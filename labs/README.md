# Writing a lab

A lab is a folder here whose name is its id. It needs four things:

| File | What it holds |
|---|---|
| `lab.toml` | id, title, level, minutes, skills, services, tasks, interview questions, and `[fidelity]` |
| `README.md` | the scenario and the tasks, written for the learner |
| `check.py` | one `check_<name>(aws)` per task; raise `labkit.runner.CheckFailed("what to fix")` when the work is not done |
| `solution/solve.py` | `solve(aws)`: a reference solution the checks are proven against |

`aws` gives boto3 clients for the local stack (`aws.sqs`, `aws.lambda_`,
`aws.athena`, ...), `aws.redis_url`, and `aws.code_dir` for labs where the
learner writes code or SQL in `answers/` (a solution points it at
`solution/answers/` instead).

## The rules every lab follows

1. **Say what is real.** `[fidelity]` lists what really runs locally
   (`real`), what is emulated or not enforced (`stand_in`), and what to expect
   on the real service (`differs_on_aws`). A learner must never walk into an
   interview believing the local stack is the cloud.
2. **Checks test behaviour, not just existence.** Where it can, a check sends
   real traffic (a message, an execution, a query) and looks at the result.
3. **Checks are proven.** `tests/labkit/test_live.py` runs every lab against
   MiniStack: on an empty stack every check fails; after `solve` every check
   passes. Add a near miss to `tests/labkit/test_near_misses.py` for the
   mistake learners make most.
4. **Skills use the names postings use**, lowercase (`aws lambda`, `redis`),
   so `labs suggest` can match them; add aliases in `labkit/suggest.py`.
5. **Original content only.** Scenarios, data and code are written for this
   repo. No copied tutorials, no real company or personal data.

Run the lab tests with `uv run pytest tests/labkit` while `docker compose up -d`
is running.
