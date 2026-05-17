"""QA harness: end-to-end checks against a real `lilbee serve`.

Two suites live here:

* ``test_opencode_matrix.py`` drives a real ``opencode`` binary against
  the OpenAI-shaped ``/v1`` surface. Gated behind ``LILBEE_QA_OPENCODE=1``
  and the ``opencode`` binary being on ``PATH``; default CI skips it.
* ``test_protocol_smoke.py`` hits ``/v1/chat/completions`` directly
  and asserts the response envelopes match the OpenAI shape. Runs
  in default CI.
"""
