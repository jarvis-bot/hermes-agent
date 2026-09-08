# Supervisor resume-request deployment (inactive template)

This architecture keeps observation separate from mutation. The deterministic
`no_agent` script only inspects a fixed board/task and appends an immutable
request. The **default gateway** re-reads task, run, Git, and candidate state and
is the only component that can accept the request and return the task to
`ready`; normal dispatch then creates one continuation run. Delegated children
remain blocked by the Kanban DB mutation guard and cannot append requests.

## Deployment order (do not skip)

1. Review and publish the exact framework commit; require focused/adjacent CI green.
2. Install that exact SHA and canary `hermes kanban list --read-only --json` on a
   disposable board. Verify DB, WAL, and SHM bytes are unchanged.
3. Generate the Cowbone manifest from
   `docs/examples/cowbone-resume-observer.manifest.json`. Before replacing any
   placeholder, re-read `t_8d82ac4d` read-only and authenticate its existing
   17-file workspace, branch, HEAD SHA, and candidate fingerprint. Never reset,
   checkout, clean, stage, or otherwise mutate that workspace.
4. Add the exact same fixed tuple under the **default profile** config only:

   ```yaml
   kanban:
     resume_request_policies:
       - board: cowbone
         task_id: t_8d82ac4d
         action: resume_iteration_budget
         workspace_path: <exact existing candidate workspace>
         branch: <exact reviewed branch>
         sha: <exact reviewed HEAD>
         candidate_fingerprint: <sha256:... from installed framework>
         block_reason_sha256: <sha256:... of the reviewed iteration-budget summary>
   ```

5. Copy the reviewed `scripts/kanban_resume_observer.py` to
   `$HERMES_HOME/scripts/kanban_resume_observer.py`, owned by the gateway account
   and not group/world writable. Place the manifest at
   `$HERMES_HOME/kanban/resume-observer.json` with mode `0600`.
6. Canary the observer once against a disposable cloned DB and copied workspace.
   A first blocker/request transition emits one JSON line; an identical second
   run emits nothing. Rejections exit/report as failures, never success.
7. Restart only the default gateway and verify it reports the exact installed
   SHA. Reviewer gateways must log that their dispatcher is disabled.
8. Create a **paused** template job, inspect it, then explicitly activate only
   after canary approval:

   ```text
   schedule: every 5m
   script: kanban_resume_observer.py
   no_agent: true
   deliver: <operator-owned alert route>
   profile: default
   initially paused: true
   ```

9. For the one-time League Table recovery, let the observer append the request;
   never manually unblock/claim it. Acceptance must create one `ready` transition
   and the ordinary default dispatcher must create exactly one new run. Keep
   recovery dependency `t_1fd4142a` untouched unless a separately reviewed policy
   explicitly supports it (this implementation does not).

## Fail-closed behavior

A request is rejected if its fixed policy is absent, status/version/block kind,
workspace path, branch, expected SHA, or candidate fingerprint changed, or if a
claim/PID/current run is present. Only `needs_input` with action
`resume_iteration_budget` is supported. Dependency, capability, credential, and
other gate failures remain untouched. Requests are content-addressed and unique;
lease fence plus one transaction makes replay safe across crashes and restarts.
