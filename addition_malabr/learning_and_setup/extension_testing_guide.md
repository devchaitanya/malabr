# Extension Testing Guide

This guide shows how to test the **active UDS extension flow** correctly and understand which code connects to which part.

Scope of this guide:
- Active path: `read_server_uds_extension` + `readServerUds` + `modelserver_uds`
- Legacy HTTP path is not the main focus here.

## 1. Architecture Map (Who Calls Whom)

Use this as the core mental model while testing.

1. Popup UI (extension)
- `src/addition_malabr/read_server_uds_extension/popup.html`
- `src/addition_malabr/read_server_uds_extension/popup.js`

2. Chromium extension API contract
- `src/extensions/common/api/read_server_uds.idl`

3. Chromium browser-side API implementation
- `src/extensions/browser/api/read_server_uds/read_server_uds_api.cc`
- `src/extensions/browser/api/read_server_uds/read_server_uds_api.h`

4. UDS transport client (C++)
- `src/extensions/browser/api/read_server_uds/ml_server_uds_v2.cc`
- `src/extensions/browser/api/read_server_uds/socket_uds.cc`

5. Python UDS server entry + routing
- `src/addition_malabr/modelserver_uds/app/server.py`
- `src/addition_malabr/modelserver_uds/app/socket_server_v2.py`
- `src/addition_malabr/modelserver_uds/app/router_v2.py`

6. Python handlers
- `src/addition_malabr/modelserver_uds/app/handler/read_data.py`
- `src/addition_malabr/modelserver_uds/app/handler/send_data.py`
- `src/addition_malabr/modelserver_uds/app/handler/qa_model.py`

7. FlatBuffer schema (extension + python payload parser)
- `src/addition_malabr/read_server_uds_extension/qa_schema.fbs`
- `src/addition_malabr/read_server_uds_extension/qa_schema_generated.js`
- `src/addition_malabr/modelserver_uds/app/handler/QAService/Payloads/*`

## 2. Labels Mapping (Critical)

These labels must match exactly from extension call -> C++ -> Python router.

- `LABEL_READ_DATA`
  - C++ sender: `read_server_uds_api.cc`
  - Python handler: `read_data.handle`

- `LABEL_SEND_DATA`
  - C++ sender: `read_server_uds_api.cc`
  - Python handler: `send_data.handle`

- `LABEL_LOAD_MODEL_BERT`
  - C++ sender: `read_server_uds_api.cc`
  - Python handler: `qa_model.load_model`

- `LABEL_INFER_MODEL_BERT`
  - C++ sender: `read_server_uds_api.cc`
  - Python handler: `qa_model.infer`

## 3. Wire Protocol You Are Testing

Every request over UDS uses this frame:

1. 4 bytes: header length (big-endian)
2. Header string: `<fb_id>,<label>,<payload_size>`
3. Payload bytes

Request reader:
- `src/addition_malabr/modelserver_uds/app/socket_server_v2.py`

Response format back to C++:
1. 4 bytes length prefix
2. UTF-8 JSON body like `{"status":"ok","message":"..."}`

Response writer:
- `src/addition_malabr/modelserver_uds/app/response.py`

## 4. Pre-Test Checklist

Run this checklist before opening Chrome.

1. Build includes custom API
- Ensure your Chromium build includes `readServerUds` API changes.

2. Debug build/run setup
- See: `src/addition_malabr/learning_and_setup/how_to_run.md`

3. Socket path is same on both sides
- C++ uses: `/tmp/malabr.sck` in `read_server_uds_api.cc`
- Python uses: `SOCKET_PATH` in `modelserver_uds/app/config.py`

4. Python dependencies available
- `torch`, `transformers`, `flatbuffers`
- See: `modelserver_uds/app/requirements.txt`

5. UDS server is running before using popup buttons

## 5. Start The Python UDS Server

Choose one way.

## Option A: Local python run

From:
- `src/addition_malabr/modelserver_uds/app`

Run:

```bash
python3 server.py
```

Expected:
- logs indicating socket server listening on `/tmp/malabr.sck`

## Option B: Container flow

Use:
- `src/addition_malabr/modelserver_uds/script.sh`

Notes:
- script mounts shared sockets and runs containerized server
- verify actual socket path aligns with C++ side

## 6. Load and Test Extension in Chromium

1. Launch Chromium with logging:

```bash
out/Default/chrome --enable-logging=stderr --v=1
```

2. Load extension folder:
- `src/addition_malabr/read_server_uds_extension`

3. Open extension popup.

4. Run tests in this order:

- Test 1: `Read Data`
- Test 2: `Send Data`
- Test 3: `Load Bert`
- Test 4: `Infer Bert`
- Test 5: `Benchmark Bert` (optional, only after inference works)

Reason for this order:
- It validates basic connection first, model load next, and heavy inference last.

## 7. Expected Result for Each Button

1. `Read Data`
- Expected message similar to alive/time response from `read_data.py`

2. `Send Data`
- Expected uppercase echo from `send_data.py`

3. `Load Bert`
- Expected success text from `qa_model.load_model`
- If already loaded, error like "Model already loaded."

4. `Infer Bert`
- Requires valid question + context
- Expected `Answer: ...`

5. `Benchmark Bert`
- Executes repeated inference calls
- Shows summary metrics in popup/alert

## 8. Code Connection Table (Quick Reference)

- Popup click handlers
  - `src/addition_malabr/read_server_uds_extension/popup.js`

- JS API declaration source
  - `src/extensions/common/api/read_server_uds.idl`

- API call implementation classes
  - `ReadServerUdsReadDataFunction`
  - `ReadServerUdsSendDataFunction`
  - `ReadServerUdsLoadModelBERTFunction`
  - `ReadServerUdsInferSingleBERTFunction`
  - file: `src/extensions/browser/api/read_server_uds/read_server_uds_api.cc`

- UDS send/receive framing
  - `src/extensions/browser/api/read_server_uds/ml_server_uds_v2.cc`

- Python label router
  - `src/addition_malabr/modelserver_uds/app/router_v2.py`

- Python model logic
  - `src/addition_malabr/modelserver_uds/app/handler/qa_model.py`

## 9. Observability: Where To Look When It Fails

1. Popup/UI errors
- extension popup text fields and error labels
- file: `read_server_uds_extension/popup.js`

2. Chromium logs
- terminal running `chrome --enable-logging=stderr --v=1`
- look for function names from `read_server_uds_api.cc`

3. Python server logs
- terminal where `server.py` runs
- check label routing, parse errors, model load errors

4. Socket mismatch symptoms
- C++ errors: connect/read/write failed
- Python shows no incoming request

## 10. Common Failure Cases and Fixes

1. Error: unknown label
- Cause: label mismatch C++ vs Python router
- Check: `read_server_uds_api.cc` and `router_v2.py`

2. Error: model not loaded
- Cause: infer called before `Load Bert`
- Fix: click `Load Bert` first

3. Error: connect failed
- Cause: python server not running or wrong socket path
- Check: `/tmp/malabr.sck`, `config.py`, `read_server_uds_api.cc`

4. Error: invalid payload type / inference parse failed
- Cause: FlatBuffer payload mismatch
- Check: `qa_schema.fbs` and generated parser classes

5. No response / hangs
- Cause: partial reads/writes or server-side exception
- Check both logs and frame parsing path in `socket_server_v2.py`

## 11. Minimal Correct Test Sequence (Recommended)

Run this exact sequence each fresh session:

1. Start Python UDS server.
2. Launch Chromium with logging.
3. Load extension folder `read_server_uds_extension`.
4. Click `Read Data` and verify success.
5. Click `Send Data` and verify uppercase output.
6. Click `Load Bert` and verify success.
7. Click `Infer Bert` with default question/context.
8. Only then run benchmark.

If all eight steps pass, your full extension pipeline is working end-to-end.

## 12. Optional Deep Validation (For Better Learning)

1. Change one label intentionally in Python router and observe failure.
2. Restore label and confirm fix.
3. Change socket path in one side only and observe connect failure.
4. Restore and confirm fix.

This makes the coupling points very clear and helps debugging faster later.

## 13. Related Docs

- Step 1 architecture doc:
  - `src/addition_malabr/learning_and_setup/extension_flow_step1.md`
- run/debug basics:
  - `src/addition_malabr/learning_and_setup/how_to_run.md`
- current high-level flow:
  - `src/addition_malabr/learning_and_setup/current_flow.md`
