# Security policy

gazefov handles two sensitive things: **infrared emitters pointed at eyes**, and **gaze data**.

* **Eye safety.** `gazefov/safety.py` is a design calculation, not a certification. Never drive IR
  LEDs near an eye without measuring irradiance at the eye position and adding hardware current
  limiting. Report any error in the safety maths as a security issue.
* **Gaze data** is biometric-adjacent. The UDP sender and the WebSocket relay bind to `127.0.0.1` by
  default and are unauthenticated; do not expose them beyond a trusted network.

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab) on
`coopersimpson101-debug/gazefov` rather than in a public issue. Supported version: the latest release.
