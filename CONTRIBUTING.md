# Contributing

We are very happy you want to contribute to RAI, and we welcome all input. This document outlines
guidelines for contributors to follow.

Our philosophy is strongly aligned with the philosophy of the ROS development process. These
guidelines have been therefore strongly influenced by the
[ROS2 Contributing document](https://docs.ros.org/en/jazzy/The-ROS2-Project/Contributing.html).

## Tenets

-   Engage Robotec.ai as early as possible

    -   Start discussions with Robotec.ai and the community early. Long time RAI contributors may have a
        clearer vision of the big picture. If you implement a feature and send a pull request without
        discussing with the community first, you are taking the risk of it being rejected, or you may be
        asked to largely rethink your design.
    -   Opening issues or using Discourse to socialize an idea before starting the implementation is
        generally preferable.

-   Adopt community best-practices whenever possible instead of ad-hoc processes

    Think about the end-users experience when developing and contributing. Features accessible to a
    larger amount of potential users, utilising widely available solutions are more likely to be
    accepted.

-   Think about the community as a whole

    Think about the bigger picture. There are developers building different robots with different
    constraints. The landscape of available AI models is rapidly changing, coming with different
    capabilities and constraints. RAI wants to accommodate requirements of the whole community.

There are a number of ways you can contribute to the RAI project.

## Discussions and support

Some of the easiest ways to contribute to RAI involve engaging in community discussions and support.
This can be done by creating Issues and RFCs on github.

## Contributing code

### Setting up the development environment

Set up your development environment following [the instructions](../../setup/install.md).

Additionally, setup the pre-commit:

```bash
sudo apt install shellcheck
pre-commit install
pre-commit run -a # Run the checks before committing
```

Optionally, install all RAI dependencies and run the tests:

```bash
poetry install --all-groups
colcon build --symlink-install
pytest tests/
```

### Starting the discussion

Always try to engage in discussion first. Browse Issues and RFCs or start a discussion on
[ROS Embodied AI Community Group Discord](https://discord.gg/3PGHgTaJSB) to see if a feature you want to propose (or a similar
one) has already been mentioned. If that is the case feel free to offer that you'll work on it, and
propose what changes/additions you will make. One of the project maintainers will assign the issue
to you, and you can start working on the code.

### Submitting code changes

To submit a change begin by forking this repository and making the changes on the fork. Once the
changes are ready to be proposed create a pull request back to the repository. In order to maintain
a linear and clear commit history please:

-   make sure that all commits have meaningful messages
-   if batches of "cleanup" or similar commits are present - squash them together
-   rebase onto the main branch of repository before making the PR

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for our commit messages. This means that each commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Common types include:

-   `feat`: A new feature
-   `fix`: A bug fix
-   `docs`: Documentation changes
-   `style`: Changes that do not affect the meaning of the code
-   `refactor`: A code change that neither fixes a bug nor adds a feature
-   `test`: Adding missing tests or correcting existing tests
-   `chore`: Changes to the build process or auxiliary tools

Always make sure that both all tests are passing before making the PR. Once this is done open your
PR describing what changes have been made and how to test if it's working. Request review from the
maintainer who assigned you the issue.

If the review requires modifications please make them on your forked repository and go through the
above process. Once the PR is accepted it will be merged into the repository. Congratulations, and
thank you for your contributions to the development of RAI.

## Versioning policy

RAI follows Semantic Versioning using the format MAJOR.MINOR.PATCH.

1. MAJOR version increments indicate incompatible API changes.
   A MAJOR release may require users to update their code, configuration, or integration logic.
   Breaking changes must be clearly documented in the release notes, including migration guidance when possible.

2. MINOR version increments indicate backward compatible functionality additions.
   A MINOR release may add new features, extensions, or optional capabilities without breaking existing behavior.
   Existing APIs must continue to work as documented.

3. PATCH version increments indicate backward compatible bug fixes.
   A PATCH release must not introduce new features or behavior changes beyond fixing defects.
   Performance improvements are acceptable if they do not change observable behavior.

For more information about semantic versioning, see [https://semver.org/](https://semver.org/).

### Pre release and development versions

During active development, pre release identifiers may be used to signal unstable or experimental states.
These versions are not guaranteed to maintain backward compatibility and should not be relied upon in production systems.

### Versioning scope

The version number applies to the public RAI API surface, including user facing Python APIs, ROS interfaces, and configuration schemas.
Internal implementation details that are not part of the documented API may change without triggering a MAJOR version bump.

### Release requirements

Before publishing a release:

1. All tests must pass.
2. Public API changes must be documented.

This policy is intended to provide clear expectations for users and contributors about compatibility and upgrade impact.
