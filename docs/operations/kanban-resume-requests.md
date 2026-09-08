# Supervisor resume requests (inactive template)

This feature is disabled by default. The observer only reads the board and writes a
one-field file to a local outbox. The default gateway authenticates that file by its
OS owner, replaces the index with its own fixed policy/current board values, and only
then appends and consumes the database request.

## Security boundary

Environment variables and Python caller identity are **not** authorization. Deployment
must use three controls:

1. a dedicated, non-login producer UID, different from the gateway and every LLM worker;
2. a root-owned observer script/manifest (`0750`/`0640` or stricter), executable only by
   that UID, with no general shell/agent job running as the producer UID; and
3. a local outbox that is not world writable. Request files are regular, single-link,
   `0600`, owned by the configured producer UID, opened with `O_NOFOLLOW`, and contain
   exact `{"policy_index": N, "state_version": V}`. `state_version` is only a replay
   fence; the configured policy supplies all authority-bearing values.

A same-UID process cannot be distinguished securely; therefore using the gateway or
agent UID as producer is rejected. Filesystem ACL/group setup must let the producer
atomically publish and the gateway remove files. If the deployment cannot provide the
dedicated UID and ownership boundary, leave the endpoint disabled.

## Configuration

Under the **default profile** only:

```yaml
kanban:
  resume_request_outbox: /var/lib/hermes-resume/outbox
  resume_request_producer_uid: 12345
  resume_request_policies:
    - board: cowbone
      task_id: t_8d82ac4d
      action: resume_iteration_budget
      workspace_path: <exact existing candidate workspace>
      branch: <exact reviewed branch>
      sha: <exact reviewed HEAD>
      candidate_fingerprint: <sha256:...>
      block_reason_sha256: <sha256:...>
      workspace_kind: dir
      # One-time opt-in only for legacy dir tasks whose branch/SHA are both NULL.
      # The gateway authenticates the exact policy-pinned bytes before binding them.
      bind_legacy_metadata: true
```

Fill `docs/examples/cowbone-resume-observer.manifest.json` from the same reviewed
facts. Keep it inactive until the framework SHA is reviewed, CI is green, the exact
SHA is installed, and a disposable-board canary passes. Restart only the default
gateway after configuration; reviewer gateways never dispatch.

## Semantics

Only `needs_input` / `resume_iteration_budget` is accepted. Policy, status, event
version, blocker digest, workspace path/kind, branch, expected SHA, canonical logical
candidate bytes, and absence of a live claim/run are revalidated. Candidate `.git`
configuration is never interpreted. A bounded request batch is leased in a short
transaction, filesystem authentication runs outside the SQLite writer lock, and a
final lease/task/policy CAS accepts or rejects it. Normal `claim_task` binds the
accepted request, prepared capability, exact task metadata, and acceptance event to a
durable run/claim before launch; concurrent ticks yield one live claim and launch.

`bind_legacy_metadata` is an explicit one-time gateway policy for the historical
`workspace_kind: dir` shape where both `branch_name` and `expected_workspace_sha` are
missing. It is not a generic unblock: the exact `needs_input` reason digest and event
version must still match. The trusted policy's branch and SHA are persisted atomically
without changing workspace bytes. New dir tasks may carry a branch only when they also
carry an authenticated expected SHA.

This is not an exactly-once external-process guarantee. A confirmed live claim prevents
concurrent launch; a failed pre-exec launch closes its run and can retry; a process that
may have launched is retried only after the existing liveness/lease reconciliation proves
it dead or expired. Workers receive the durable run and claim identities.

`kanban list --read-only` and observer inspection skip initialization, migrations, and
readiness recomputation. WAL tests keep a writer and WAL/SHM open, prove committed-WAL
freshness, and assert no board-file inventory, inode, mode, owner, size, mtime, DB bytes,
or WAL bytes change. SQLite may update volatile lock words inside the existing SHM
mapping; those are coordination state, not durable board content.

No task should be manually unblocked or claimed for one-time recovery. Keep the supplied
Cowbone and League Table workspaces untouched until separate activation approval.
