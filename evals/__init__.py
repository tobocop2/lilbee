"""Version-controlled evaluation harnesses; never shipped with the package."""

import os

# ragas posts a usage event to a third-party endpoint on every llm_factory and
# evaluate call. Opt out at import: a benchmark run should not make outbound
# calls the manifest does not declare, and the pods this runs on are often
# firewalled, where the post is a timeout on the critical path rather than a
# no-op. setdefault, so an operator who wants it can still turn it back on.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
