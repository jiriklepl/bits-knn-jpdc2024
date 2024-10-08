#!/bin/bash

# find potential updates for all packages in external/

set -e

PREFER_TAGS=${PREFER_TAGS:-1}

EXTERNAL_DIR=${EXTERNAL_DIR:-external}

if ! [ "$PREFER_TAGS" -eq 0 ]; then
    PREFER_TAGS=1
fi

if ! which awk >/dev/null; then
    echo "awk not found, please install it"
    exit 1
fi

if ! which git >/dev/null; then
    echo "git not found, please install it"
    exit 1
fi

if ! which realpath >/dev/null; then
    echo "realpath not found, please install it"
    exit 1
fi

echo "Checking for updates to this repository and $EXTERNAL_DIR/* (PREFER_TAGS=$PREFER_TAGS)"

git fetch --quiet
git submodule update --init --recursive --quiet

EXTERNAL_DIR=$(realpath external)
for package in "$(realpath .)" "$EXTERNAL_DIR"/*; do
    if [ -d "$package" ]; then
        if [ -f "$package/.git" ] || [ -d "$package/.git" ]; then
            git -C "$package" fetch --quiet
            git -C "$package" fetch --tags --quiet

            REMOTE_CODENAME=$(git -C "$package" remote -v | awk '/fetch/{print $1; exit}')

            # get HEAD commit hash
            HEAD=$(git -C "$package" rev-parse HEAD)
            # get remote HEAD commit hash

            if git -C "$package" rev-parse "$REMOTE_CODENAME/HEAD" >/dev/null 2>&1; then
                REMOTE_HEAD=$(git -C "$package" rev-parse "$REMOTE_CODENAME/HEAD")
                REMOTE_NAME="$REMOTE_CODENAME/HEAD"
            elif git -C "$package" rev-parse "$REMOTE_CODENAME/master" >/dev/null 2>&1; then
                REMOTE_HEAD=$(git -C "$package" rev-parse "$REMOTE_CODENAME/master")
                REMOTE_NAME="$REMOTE_CODENAME/master"
            elif git -C "$package" rev-parse "$REMOTE_CODENAME/main" >/dev/null 2>&1; then
                REMOTE_HEAD=$(git -C "$package" rev-parse "$REMOTE_CODENAME/main")
                REMOTE_NAME="$REMOTE_CODENAME/main"
            else
                REMOTE_HEAD=$(git -C "$package" rev-parse "$REMOTE_CODENAME/FETCH_HEAD")
                REMOTE_NAME="$REMOTE_CODENAME/FETCH_HEAD"
            fi

            HEAD_TAG=$(git -C "$package" tag --points-at HEAD | awk '{print $0; exit}')

            if [ "$HEAD" != "$REMOTE_HEAD" ] || [ -z "$HEAD_TAG" ]; then
                # count commits behind/ahead
                COMMITS=$(git -C "$package" rev-list --count HEAD.."$REMOTE_HEAD")
                if [ "$PREFER_TAGS" -eq 0 ] || [ -z "$(git -C "$package" tag)" ]; then
                    if [ "$HEAD" != "$REMOTE_HEAD" ]; then
                        echo "Potential update for $package: $HEAD -> $REMOTE_HEAD ($COMMITS commits ahead of $(git -C "$package" tag --points-at HEAD | awk '{print "tag " $1 " (HEAD)";succ=1;exit}END{if (!succ) print "HEAD"}'))"
                        echo "  switch to $REMOTE_NAME? [y/N]"
                        read -r response </dev/tty
                        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
                            git -C "$package" checkout "$REMOTE_NAME" && echo "  Updated to $REMOTE_NAME"
                        fi
                    else
                        echo "No updates for $package (HEAD at $HEAD)"
                    fi
                else
                    echo "Looking for tags for $package (HEAD$(test -n "$HEAD_TAG" && echo " (tag $HEAD_TAG)") at $HEAD)"

                    if git -C "$package" tag >/dev/null; then
                        TAGS=0
                        SAME_TAGS=0
                        for tag in $(git -C "$package" tag --contains HEAD --sort=-creatordate); do
                            COMMIT=$(git -C "$package" rev-list -n 1 "$tag")
                            TAGS=$((TAGS + 1))
                            if [ "$COMMIT" != "$HEAD" ]; then
                                echo "  $tag ($(git -C "$package" rev-list --count HEAD.."$tag") commits ahead of HEAD; $(git -C "$package" diff --shortstat HEAD "$tag"))"

                                echo "  switch to $tag? [y/N]"
                                read -r response </dev/tty
                                if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
                                    git -C "$package" checkout "$tag" && echo "  Updated to $tag"
                                    break
                                fi
                            else
                                SAME_TAGS=$((SAME_TAGS + 1))
                            fi
                        done

                        if [ "$TAGS" -eq "$SAME_TAGS" ] && [ "$TAGS" -ne 0 ]; then
                            echo "  No newer tags available"
                        elif [ "$TAGS" -eq 0 ]; then
                            # try past tags
                            for tag in $(git -C "$package" tag --sort=-creatordate); do
                                COMMIT=$(git -C "$package" rev-list -n 1 "$tag")
                                TAGS=$((TAGS + 1))
                                if [ "$COMMIT" != "$HEAD" ]; then
                                    echo "  $tag ($(git -C "$package" rev-list --count "$tag"..HEAD) commits behind of HEAD; $(git -C "$package" diff --shortstat HEAD "$tag"))"

                                    echo "  switch to $tag? [y/N]"
                                    read -r response </dev/tty
                                    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
                                        git -C "$package" checkout "$tag" && echo "  Downgraded to $tag"
                                        break
                                    fi
                                fi
                            done
                        fi

                        if [ "$TAGS" -eq 0 ]; then
                            echo "  No tags available"
                        fi
                    fi
                fi
            else
                echo "No updates for $package"
            fi
        else
            echo "Skipping $package" # possible extension: check for updates in other ways
        fi
    fi
done
