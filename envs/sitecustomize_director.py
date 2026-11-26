# Added 2026-11-03 for the DIRECTOR reproduction: /tmp sits on the nearly full root disk.
import os
if os.path.isdir("/ephemeral/trinhpn_uq/tmp"):
    os.environ["TMPDIR"] = "/ephemeral/trinhpn_uq/tmp"
