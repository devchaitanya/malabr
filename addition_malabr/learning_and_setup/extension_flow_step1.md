# Extension Flow (Step 1)

This document explains the **current extension architecture** in `addition_malabr`:
- where the extension is created,
- what it does,
- how it does it,
- and how data flows from popup input to model output.

## 1. What This Extension Is

The active extension is a Chromium extension that calls a **custom Chromium API** named `readServerUds`.

It is not calling a public Chrome API. Instead, it uses custom APIs added in this codebase:
- `chrome.readServerUds.*` (active UDS flow)
- `chrome.readServer.*` (older HTTP flow)

The extension UI sends request data to Chromium C++ code, which then talks to a Python model server over a Unix Domain Socket (UDS).

## 2. Where The Extension Is Created

## Active extension (UDS)
- `src/addition_malabr/read_server_uds_extension/manifest.json`
- `src/addition_malabr/read_server_uds_extension/popup.html`
- `src/addition_malabr/read_server_uds_extension/popup.js`
- `src/addition_malabr/read_server_uds_extension/qa_schema.fbs`
- `src/addition_malabr/read_server_uds_extension/qa_schema_generated.js`

This is the extension you use for the current UDS model flow.

## Older extension (HTTP)
- `src/addition_malabr/reading server extension/manifest.json`
- `src/addition_malabr/reading server extension/popup.js`

This one uses `chrome.readServer.*` and HTTP (`localhost:5000`).

## 3. Where The Custom Chromium API Is Defined

The extension API is defined in Chromium IDL + C++ implementation:

- IDL contract:
  - `src/extensions/common/api/read_server_uds.idl`
- Browser-side API implementation:
  - `src/extensions/browser/api/read_server_uds/read_server_uds_api.h`
  - `src/extensions/browser/api/read_server_uds/read_server_uds_api.cc`
- UDS transport client:
  - `src/extensions/browser/api/read_server_uds/ml_server_uds_v2.h`
  - `src/extensions/browser/api/read_server_uds/ml_server_uds_v2.cc`
  - `src/extensions/browser/api/read_server_uds/socket_uds.h`
  - `src/extensions/browser/api/read_server_uds/socket_uds.cc`

So the extension call path is:
1. Popup JS calls `chrome.readServerUds.*`.
2. Chromium IDL binding validates and maps args.
3. C++ API class sends framed data over UDS.

## 4. What The Extension Does

The popup exposes the following operations (active extension):

- `readData`:
  - quick health check path (`LABEL_READ_DATA`)
- `sendData`:
  - test payload path (`LABEL_SEND_DATA`)
- `loadModelBERT`:
  - load QA model on Python side (`LABEL_LOAD_MODEL_BERT`)
- `inferSingleBERT`:
  - send FlatBuffer payload with `question` + `context` (`LABEL_INFER_MODEL_BERT`)
- benchmark button:
  - repeatedly invokes `inferSingleBERT` and measures latency/throughput

Primary runtime file:
- `src/addition_malabr/read_server_uds_extension/popup.js`

## 5. How It Does It (Protocol + Routing)

## 5.1 Payload format from extension
For inference, popup code builds a FlatBuffer object:
- schema: `qa_schema.fbs`
- root table includes union payload (`QARequest`)
- file identifier: `QASV`

## 5.2 Wire protocol (C++ <-> Python over UDS)
C++ sends:
1. 4-byte big-endian header length
2. Header string: `<fb_id>,<label>,<payload_size>`
3. Raw payload bytes

Python server reads exactly this format, then routes by `label`.

## 5.3 Python server side
Main files:
- `src/addition_malabr/modelserver_uds/app/server.py`
- `src/addition_malabr/modelserver_uds/app/socket_server_v2.py`
- `src/addition_malabr/modelserver_uds/app/router_v2.py`
- `src/addition_malabr/modelserver_uds/app/handler/qa_model.py`

Handlers:
- `LABEL_READ_DATA` -> `read_data.handle`
- `LABEL_SEND_DATA` -> `send_data.handle`
- `LABEL_LOAD_MODEL_BERT` -> `qa_model.load_model`
- `LABEL_INFER_MODEL_BERT` -> `qa_model.infer`

Inference handler behavior:
1. Parse FlatBuffer payload into `(question, context)`.
2. Run HuggingFace QA pipeline (`csarron/mobilebert-uncased-squad-v2`).
3. Return JSON response with length prefix.

## 6. End-to-End Data Flow (Single Inference)

1. User enters question/context in popup.
2. `popup.js` serializes into FlatBuffer (`QASV`).
3. `chrome.readServerUds.inferSingleBERT(...)` is invoked.
4. Chromium C++ API function receives request.
5. C++ UDS client writes framed payload to `/tmp/malabr.sck`.
6. Python UDS server accepts socket, parses header + payload.
7. Router maps `LABEL_INFER_MODEL_BERT` to inference handler.
8. Handler runs model and creates JSON response.
9. Python sends response with 4-byte length prefix.
10. C++ reads response and returns to extension callback.
11. Popup displays answer/error in UI.

## 7. Runtime Components You Need

- Chromium build with your custom extension API (`readServerUds`) compiled in.
- Python UDS server running and listening on socket path:
  - `src/addition_malabr/modelserver_uds/app/config.py`
  - default: `/tmp/malabr.sck`
- Extension loaded from:
  - `src/addition_malabr/read_server_uds_extension`

## 8. Active vs Legacy Path

## Active (recommended)
- Extension: `read_server_uds_extension`
- API: `readServerUds`
- Transport: UDS
- Backend: `modelserver_uds`

## Legacy
- Extension: `reading server extension`
- API: `readServer`
- Transport: HTTP
- Backend: `modelserver` (Flask/Quart variants)

## 9. Notes for Step 1

This file is your **Step 1 architecture reference**. Before any change:
1. Confirm you are using `read_server_uds_extension` and not the legacy extension.
2. Confirm UDS socket path matches on both C++ and Python sides.
3. Confirm labels are identical between C++ sender and Python router.
4. Confirm FlatBuffer file identifier remains `QASV` on both sides.

If any one of these does not match, requests fail even when all code compiles.
