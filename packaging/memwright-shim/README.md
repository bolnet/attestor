# memwright (deprecated shim)

`memwright` has been renamed to **[`attestor`](https://pypi.org/project/attestor/)**.

This package is a thin compatibility shim. Installing it pulls in `attestor` as a dependency and re-exports its public API (`AgentMemory`, `Memory`, `RetrievalResult`) under the `memwright` namespace. Importing `memwright` emits a `DeprecationWarning`.

## Migrate

```bash
pip uninstall memwright
pip install attestor
```

```diff
- from memwright import AgentMemory
+ from attestor import AgentMemory
```

This shim will be removed in **v3.2**. See the [Attestor CHANGELOG](https://github.com/bolnet/attestor/blob/main/CHANGELOG.md) for the full migration guide.
