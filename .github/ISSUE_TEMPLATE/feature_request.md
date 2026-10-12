name: Feature Request
description: Suggest a capability for Assembly
title: "[feature]: "
labels: ["enhancement"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem
      description: What are you trying to do that is hard today? Describe the problem, not the solution.
    validations:
      required: true

  - type: textarea
    id: proposal
    attributes:
      label: Proposed change
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered

  - type: checkboxes
    id: checks
    attributes:
      label: Checks
      options:
        - label: I searched existing issues and discussions for this
          required: true
