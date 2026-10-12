# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Instead, report it
privately via email to devcartofell@gmail.com, or use GitHub's private
vulnerability reporting on this repository:

1. Go to the **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the issue, the affected version, and how to reproduce it.

We will acknowledge the report as soon as we can and coordinate a fix and
disclosure with you.

## Supported versions

Assembly is pre-1.0 and only the latest release is supported with security
fixes.

## Scope notes

`assembly serve` binds to loopback by default. Binding it to a non-loopback
address requires an API key, and the server confines `/api/load` to the served
archive's directory. Reports about the server's threat model are welcome, but
please read the README first for what is and is not defended against.
