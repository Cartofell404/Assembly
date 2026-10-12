name: Bug Report
description: Report a bug in Assembly
title: "[bug]: "
labels: ["needs triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for filing an issue. Please search existing issues first.

  - type: input
    id: version
    attributes:
      label: Assembly version
      description: Output of `assembly --version`
      placeholder: "assembly 1.0.0-beta.1"
    validations:
      required: true

  - type: dropdown
    id: kind
    attributes:
      label: Type
      options:
        - Bug
        - Feature request
        - Documentation
        - Performance
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: What happened
      description: What did you expect, and what happened instead?
    validations:
      required: true

  - type: textarea
    id: repro
    attributes:
      label: Reproduction
      description: Minimal command sequence that reproduces the problem.
      render: shell
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: |
        - Assembly version (`assembly --version`):
        - OS and architecture:
        - How Assembly was installed (cargo, installer script, prebuilt binary):
        - GPU / accelerator:
        - Assembly mode: `run` or `serve`
    validations:
      required: true

  - type: checkboxes
    id: checks
    attributes:
      label: Checks
      options:
        - label: I searched existing issues for this
          required: true
        - label: I removed secrets, API keys, and personal data from this report
          required: true
