"""Chat template application and UTF-8-safe streaming -- sections 6c, 6d.

See design_doc/implementation_notes.md, 'Code rationale: formatter'."""

import llama_cpp.llama_cpp as C


class TemplateError(RuntimeError):
    """The model's chat template does not support incremental turn building."""


class ChatFormatter:
    """Builds prompt tokens for a session, one turn at a time (section 6c).

    Sessions keep their KV cache resident so each turn prefills only the NEW
    tokens. Re-rendering and re-prefilling the whole history every turn would
    defeat that entirely. So this class tracks exactly what text is already
    represented in the KV and hands back only the delta.

    Model-agnostic BY CONSTRUCTION. An earlier draft of section 6c specified
    appending a hardcoded '<|im_start|>{role}\\n{content}<|im_end|>\\n' fragment.
    That is ChatML, i.e. Qwen-specific -- it would silently produce malformed
    structure on gemma-3, whose template uses <start_of_turn>. Instead the delta
    is computed FROM the model's own template: render the full conversation, and
    take the suffix past what is already rendered. The template is the single
    source of truth and no marker is hardcoded anywhere.

    Why the delta can be tokenized on its own, which is the non-obvious part:
    BPE merges across concatenation boundaries in general -- measured,
    tok("hell")+tok("o") = [56095, 78] but tok("hello") = [14990]. Tokenizing a
    delta separately would therefore normally be unsafe. It is safe here only
    because every turn boundary begins at a SPECIAL token (the turn terminator),
    and special tokens are hard boundaries the BPE merge never crosses --
    verified: splitting 'I am Qwen.<|im_end|>' at the special token yields
    identical ids. verify_against_full() re-checks this property rather than
    trusting it.
    """

    def __init__(self, model, vocab, disable_thinking=True):
        self._model = model
        self._vocab = vocab
        tmpl = C.llama_model_chat_template(model, None)
        if not tmpl:
            raise TemplateError("model carries no chat template")
        self._tmpl = tmpl
        self._messages = []      # full logical history, for re-rendering
        self._rendered = ""      # EXACTLY the text currently represented in KV
        self._think_suffix = (self._extract_no_think_suffix(tmpl)
                              if disable_thinking else "")

    @staticmethod
    def _extract_no_think_suffix(tmpl):
        """The text this model's template appends to switch reasoning OFF.

        Reasoning models emit a visible chain of thought before the answer.
        That is not merely noise in the UI: those tokens are generated, so they
        consume the per-request output cap, the session's context budget and
        wall-clock time. Suppressing them downstream (in the streamer or the
        page) would pay all three costs and then throw the result away.
        Suppressing them at the prompt means they are never produced.

        Qwen3's template does this with
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
        i.e. it pre-fills an already-closed think block so the model treats
        reasoning as finished. llama_chat_apply_template cannot pass template
        VARIABLES, only messages, so the suffix is READ OUT of the template
        source instead of hardcoded -- a model whose template has no such
        branch simply gets "" and is unaffected.
        """
        try:
            source = tmpl.decode("utf-8") if isinstance(tmpl, bytes) else tmpl
        except (UnicodeDecodeError, AttributeError):
            return ""
        import re
        m = re.search(
            r"enable_thinking\s+is\s+false\s*%\}\s*\{\{-?\s*'((?:[^'\\]|\\.)*)'",
            source)
        if not m:
            return ""
        # The captured text is a Jinja string literal: unescape it the same way
        # Jinja would, so '\n' becomes a real newline.
        return m.group(1).encode().decode("unicode_escape")

    # -- rendering ------------------------------------------------------------

    def _apply(self, messages, add_generation_prompt):
        import ctypes
        arr = (C.llama_chat_message * len(messages))()
        keep = []                      # keep byte objects alive past the loop
        for i, (role, content) in enumerate(messages):
            rb, cb = role.encode(), content.encode()
            keep.append((rb, cb))
            arr[i].role = ctypes.c_char_p(rb)
            arr[i].content = ctypes.c_char_p(cb)
        size = 65536
        while True:
            buf = ctypes.create_string_buffer(size)
            n = C.llama_chat_apply_template(
                self._tmpl, arr, len(messages), add_generation_prompt, buf, size)
            if n < 0:
                raise TemplateError(f"llama_chat_apply_template failed ({n})")
            if n <= size:
                return buf.raw[:n].decode("utf-8")
            size = n + 1               # documented API: retry with the needed size

    def _tokenize(self, text, parse_special=True, add_special=False):
        import ctypes
        b = text.encode("utf-8")
        if not b:
            return []
        cap = len(b) + 64
        buf = (C.llama_token * cap)()
        # add_special controls BOS. llama_chat_apply_template renders the
        # template TEXT but never emits the actual BOS token bytes ({{ bos_token
        # }} is a HuggingFace-ism its engine drops), so BOS has to come from
        # here -- and only on the very first delta of a conversation. This still
        # honours the model's own tokenizer.ggml.add_bos_token flag: it is a
        # no-op for models like Qwen3 that set it false, and prepends <bos> for
        # models like gemma-3 that require it. Without it gemma-3 loses track of
        # who is speaking and answers as if it were the user.
        n = C.llama_tokenize(self._vocab, b, len(b), buf, cap,
                             add_special,
                             parse_special)
        if n < 0:
            raise TemplateError(f"llama_tokenize overflow ({n})")
        return list(buf[:n])

    # -- the two operations the engine actually calls --------------------------

    def user_turn(self, text):
        """Record a user message and return ONLY the new tokens to prefill."""
        first_turn = not self._rendered
        self._messages.append(("user", text))
        full = self._apply(self._messages, True)

        # Load-bearing invariant, checked rather than assumed: everything already
        # in the KV must still be a prefix of the newly rendered conversation. If
        # a template ever violates this (a summarising or history-rewriting
        # template would), incremental prefill is invalid and the KV would
        # silently disagree with what the model thinks it has seen. Fail loudly
        # instead -- this is the "subtly wrong output" class, not a crash class.
        if not full.startswith(self._rendered):
            raise TemplateError(
                "template broke the prefix property; incremental prefill is "
                "unsafe for this model")

        # Append the model's own "reasoning off" marker after the generation
        # prompt, exactly where its template would have put it.
        full += self._think_suffix
        delta = full[len(self._rendered):]
        self._rendered = full
        # Only the first delta of the conversation carries BOS (see _tokenize).
        return self._tokenize(delta, add_special=first_turn)

    def assistant_generated(self, text):
        """Record what the model actually produced.

        The generated tokens are already in the KV, so this adds no tokens. It
        only keeps _rendered in step with reality. Note the KV does NOT contain
        the turn terminator: generation stops AT the end-of-generation token and
        that token is never decoded. The next user_turn's delta therefore begins
        with the terminator, which is exactly what makes that delta start on a
        special token.
        """
        # The no-think marker is recorded as part of what the assistant said,
        # because in KV terms that is exactly what it is: those tokens sit
        # inside the assistant block, ahead of the generated text. Storing it
        # here rather than special-casing every render keeps _apply()'s output
        # matching the cache on the NEXT turn -- an earlier version appended it
        # only to _rendered, so the re-render no longer had it and the prefix
        # check failed on turn two.
        self._messages.append(("assistant", self._think_suffix + text))
        self._rendered += text

    def _expected_render(self):
        """The text that SHOULD currently be in the KV.

        Phase-dependent, and the two cases are legitimately different -- an
        earlier version compared against the wrong one and reported a false
        divergence. After a completed assistant turn the KV holds the full
        render up to and including the user turn plus the generated text, but
        NOT the turn terminator (generation stops AT the end-of-generation
        token and never decodes it) and NOT the next generation prompt.
        """
        if not self._messages:
            # An empty conversation means an empty KV. The template still emits
            # a bare generation prompt ('<|im_start|>assistant\n' on Qwen3) for
            # zero messages, and returning that would claim content the cache
            # does not hold -- which breaks the prefix check on the very next
            # turn. Rolling a session's first turn back is exactly this case.
            return ""
        if self._messages[-1][0] == "assistant":
            return self._apply(self._messages[:-1], True) + self._messages[-1][1]
        return self._apply(self._messages, True) + self._think_suffix

    def drop_messages(self, start, count):
        """Remove messages that section 8 compaction evicted from the KV.

        Section 8 specifies compaction entirely in terms of KV positions and
        never mentions this. That is the same gap section 6b had: the formatter
        is a SECOND per-session state holding the conversation, and a KV
        mutation that does not update it leaves _rendered describing content the
        model no longer has. The next turn is then built against a conversation
        that does not exist.
        """
        del self._messages[start:start + count]
        self._rendered = self._expected_render()

    # -- checkpoint / restore, for section 6b's rollback -----------------------

    def checkpoint(self):
        """Capture enough state to undo an abandoned turn.

        Section 6b specifies rollback purely as KV positions. It predates
        section 6c, which introduced a SECOND piece of per-session state -- the
        rendered conversation -- that must roll back in lockstep. Rolling back
        one without the other desyncs the formatter from the KV, and the next
        turn is built on a conversation the model never saw.

        A MESSAGE COUNT, not a (count, rendered_length) pair. The length form
        was wrong for the same reason section 8 gives about absolute positions:
        a compaction firing while the checkpoint is live shortens _rendered
        underneath it, and restoring to the old length then leaves the
        formatter describing text that no longer exists. A count survives that,
        because compact() can shift it exactly as it shifts every position.
        """
        return len(self._messages)

    def restore(self, cp):
        del self._messages[int(cp):]
        # Recompute rather than truncate: _expected_render() already knows what
        # the KV holds for each phase, so this cannot drift out of step with it.
        self._rendered = self._expected_render()

    # -- self-check for the section 12 harness ---------------------------------

    def verify_against_full(self):
        """Assert incremental building matches a from-scratch render, in TOKENS.

        Section 6c only ever compared strings. The KV holds tokens, and BPE can
        merge across boundaries, so string equality does not imply token
        equality. This checks the property that actually matters.
        """
        truth = self._tokenize(self._expected_render())
        rebuilt = self._tokenize(self._rendered)
        if truth != rebuilt:
            raise TemplateError(
                f"incremental render diverged from full render "
                f"({len(rebuilt)} tokens vs {len(truth)})")
        return True


def is_stop_token(vocab, token):
    """True if generation must stop at this token.

    Uses llama_vocab_is_eog, NOT a comparison against llama_vocab_eos. Measured
    on Qwen3-0.6B: eos() returns only <|im_end|> (151645), but SIX tokens are
    flagged end-of-generation -- </s>, <|endoftext|>, <|im_end|>, <|fim_pad|>,
    <|repo_name|>, <|file_sep|>. Checking eos alone would miss five of them and
    the model would run on until the output cap instead of ending the turn.
    """
    return bool(C.llama_vocab_is_eog(vocab, token))


# ---------------------------------------------------------------------------
# Section 6d -- UTF-8 safe streaming
# ---------------------------------------------------------------------------

class Utf8Streamer:
    """Buffers token bytes so no frame ever splits a UTF-8 character.

    Measured, not hypothetical: scanning the first 20,000 vocabulary entries,
    282 tokens (~1.4%) produce bytes that are NOT valid UTF-8 standalone -- e.g.
    token 94 -> b'\\xa1', one piece of a multi-byte character from BPE's
    byte-level fallback. Emitting each token's bytes as produced means any
    non-English text, many symbols, or emoji eventually splits a character
    across two frames, and the browser-side decoder shows a replacement
    character or throws.

    So: accumulate bytes, emit only the prefix that decodes cleanly, and hold
    the trailing partial bytes for the next token.
    """

    # A UTF-8 character is at most 4 bytes, so a legitimate partial sequence can
    # never exceed 3 held bytes. Anything longer is genuinely malformed output
    # rather than an incomplete character, and holding it forever would stall
    # the stream silently -- flush it lossily instead of hanging.
    MAX_HELD_BYTES = 3

    def __init__(self):
        self._buf = bytearray()

    def push(self, raw):
        """Add token bytes; return the text safe to send now (may be '')."""
        self._buf.extend(raw)
        # Find the longest prefix that is valid UTF-8. Decoding the whole buffer
        # and catching the error tells us exactly where the good prefix ends,
        # which is cheaper and more precise than scanning byte patterns.
        try:
            text = self._buf.decode("utf-8")
            self._buf.clear()
            return text
        except UnicodeDecodeError as e:
            good = self._buf[:e.start].decode("utf-8")
            held = self._buf[e.start:]
            if len(held) > self.MAX_HELD_BYTES:
                # Not a partial character -- genuinely invalid bytes. Emit a
                # replacement rather than holding them forever.
                good += held.decode("utf-8", "replace")
                held = bytearray()
            self._buf = bytearray(held)
            return good

    def flush(self):
        """End of stream: emit whatever is left, lossily if incomplete.

        Trailing bytes here mean generation stopped mid-character. Showing a
        replacement character is more honest than silently dropping output.
        """
        if not self._buf:
            return ""
        text = self._buf.decode("utf-8", "replace")
        self._buf.clear()
        return text
