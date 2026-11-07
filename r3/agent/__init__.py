"""Package agent is the conversational side of the Controller: the intent
router that detects when a message asks for a workflow (``router``), the
advisor that answers everything else with read-only tools (``advisor``),
the step that turns a route into a panel action (``actions``), the model
registry (``models``) and the SSE event contract (``events``)."""
