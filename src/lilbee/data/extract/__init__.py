"""Document extraction and chunking.

Everything that turns a raw file into embeddable text lives here: the bridge to
xberg's ``extract`` (:mod:`.xberg`), the ingest-facing document/markdown drivers
(:mod:`.document`), batching (:mod:`.batch`), tree-aware chunking (:mod:`.chunk`,
:mod:`.code_chunker`) and the plugin backends that route OCR, embedding and
tokenization through the fleet (:mod:`.backends`). Corpus orchestration (the
sync pipeline, discovery, workers) stays in :mod:`lilbee.data.ingest`.
"""
