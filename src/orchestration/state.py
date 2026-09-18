"""Shared state models for the AgentHive LangGraph workflow."""

from typing import Any, Literal, TypedDict


# All four domain specialists that exist in the system.
AgentName = Literal["research", "coder", "data", "writer"]

TaskStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "blocked",
]

ReviewDecision = Literal[
    "approved",
    "retry",
    "escalate",
]


class SubTask(TypedDict, total=False):
    """A single executable unit created by the supervisor."""

    id: str
    title: str
    description: str
    assigned_agent: AgentName
    dependencies: list[str]          # list of SubTask ids that must complete first
    instruction: str                  # what the specialist must do
    expected_output: str              # what a good result looks like
    status: TaskStatus
    output: str
    error: str
    confidence: float
    retry_count: int


class TaskPlan(TypedDict, total=False):
    """Structured execution plan created by the supervisor."""

    objective: str
    reasoning: str
    subtasks: list[SubTask]
    execution_order: list[str]        # ordered list of SubTask ids
    status: TaskStatus


class ToolCallLog(TypedDict, total=False):
    """Audit record generated whenever a registered tool is invoked."""

    call_id: str
    timestamp: str
    tool_name: str
    agent_name: str
    subtask_id: str
    arguments: dict[str, Any]
    result_preview: str
    success: bool
    error: str
    duration_ms: float


class ReviewResult(TypedDict, total=False):
    """Structured decision returned by the reviewer."""

    decision: ReviewDecision
    approved: bool
    confidence: float
    feedback: str
    issues: list[str]
    retry_agent: AgentName
    retry_subtask_id: str


class EscalationDetails(TypedDict, total=False):
    """Information packaged when human intervention is required."""

    required: bool
    reason: str
    severity: Literal["low", "medium", "high"]
    requested_action: str
    approval_level: Literal["Notify", "Approve Action", "Approve Plan", "Take Over"]
    trigger_type: Literal[
        "low_plan_confidence",
        "sensitive_operation",
        "explicit_user_request",
        "reviewer_escalation",
        "retries_exhausted",
    ]
    context: dict[str, Any]


class AgentHiveState(TypedDict, total=False):
    """
    Complete state shared by the LangGraph nodes.

    Every agent returns only the fields it changes. LangGraph merges those
    updates into this state while the workflow is running.
    """

    # Original user request and context
    task: str
    user_id: str
    task_id: str
    db_path: str                         # Optional path override for database checkpointer/queue
    pause_mode: str                      # e.g. "step_by_step" for explicit user pause request
    auto_approve_remaining: bool         # True if user clicked "Approve Plan" to skip subsequent pauses

    # Planning
    task_plan: TaskPlan
    selected_agent: AgentName
    current_subtask_id: str
    current_subtask: SubTask
    pending_subtask_ids: list[str]
    completed_subtask_ids: list[str]

    # Specialist execution
    specialist_output: str
    specialist_outputs: dict[str, str]   # subtask_id -> output
    specialist_confidence: float
    specialist_error: str

    # Tool execution audit trail
    tool_logs: list[ToolCallLog]
    tools_used: list[str]

    # Review and retry control
    review: ReviewResult
    retry_count: int
    max_retries: int
    low_confidence_threshold: float

    # Human escalation
    escalation: EscalationDetails
    needs_human_review: bool

    # Persistent memory
    memory_context: str
    memory_id: str

    # Final workflow result
    final_answer: str
    workflow_status: Literal[
        "planning",
        "executing",
        "reviewing",
        "retrying",
        "escalated",
        "completed",
        "failed",
    ]

    # Human-readable execution history
    trace: list[str]
