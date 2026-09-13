# Security policy

Report vulnerabilities privately to the repository maintainers. Do not include
real credentials, private video, dataset participant IDs, or internal paths in
an issue or proof of concept.

Supported releases receive fixes on the latest minor version. The viewer is a
local research tool: it has no built-in authentication and must not be exposed
directly to the public internet. Use localhost or an authenticated TLS reverse
proxy.

If a credential is discovered in source or history, treat it as compromised,
rotate it immediately, and remove it from both the current tree and Git history.

