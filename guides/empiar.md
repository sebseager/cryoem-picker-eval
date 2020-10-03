# EMPIAR

## Introduction

[EMPIAR](https://www.ebi.ac.uk/pdbe/emdb/empiar/) (the Electron Microscopy Public Image Archive) is an online database 
provided by EMBL-EBI. It contains several hundred (as of October 1, 2020) electron microscopy data sets, which can be 
downloaded for use in microscopy-based research.

EMPIAR entries are organized by accession code, which are of the format `EMPIAR-#####`. For now, the `#####` portion 
starts with a `1` followed by zeros padding the data set number, e.g., `EMPIAR-10017`. Each EMPIAR entry also has a DOI 
number of the format `https://dx.doi.org/10.6019/EMPIAR-#####`, which should be used when citing data.

This document describes how to transfer these data to a local machine or computing cluster for analysis. Since these 
data sets can be between hundreds of gigabytes and several terabytes in size, it is recommended to use a machine or 
cluster node with a dedicated high-bandwidth network connection for transfers.

## Method 1: Direct FTP Download

This method performs a direct download operation using `wget`. It can therefore fail if the recipient machine 
experiences power loss or network timeout, or if the `wget` process or its containing shell session is interrupted. 
To protect transfers against these last two possibilities, we suggest using `screen` (manual page 
[here](https://linux.die.net/man/1/screen)) or `tmux` (manual page [here](https://linux.die.net/man/1/tmux)) if 
appropriate for your computing setup. See the linked manual pages or the 
[Protecting Transfers with `tmux`](#protecting-transfers-with-tmux) section for more information.

To download a complete EMPIAR entry in the current working directory, run the following, replacing `#####` with the 
appropriate EMPIAR accession code. Note that the final `/` is important when using `wget`—otherwise, it will try to 
interpret `#####` as a file instead of a directory.

```shell script
wget ftp://ftp.ebi.ac.uk/empiar/world_availability/#####/
```

Note: the port is `22`, user is `anonymous` (or `Guest` on macOS), and password is left blank (although `wget` 
should identify these by default).

## Method 2: Globus

For organizations that have computing clusters with a Globus endpoint, it is also possible to use the Globus service 
to transfer EMPIAR data much more rapidly, and without risk of interruption. These transfers can be managed from the 
Globus web interface, as follows, and do not rely on a shell session.

1. Towards the bottom of an EMPIAR entry page (e.g., https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10017/) click the 
`Browse Globus` link
2. Enter the name of your institution and follow login prompts
3. The File Manager should now be displayed; the Collection bar should say `Shared EMBL-EBI public endpoint` and the 
Path bar should say something like `/gridftp/empiar/world_availability/#####`
4. If the File Manager is not already in double-pane view, click `Transfer or Sync to...` in the menu at right (has an 
icon of two arrows pointing in opposite directions)
5. Click the search bar above the right-hand pane and enter the name of your organization's Globus endpoint
6. The right-hand pane should now populate with user-specific contents of the endpoint you specified; navigate to a 
suitable download location
7. Select files and/or directories to transfer in the left hand pane, then click `Start >` to download them to the 
destination specified in the right-hand Path bar

Transfer progress is displayed in the Activity tab in the menu at left. The transfer will be scheduled by Globus, 
so the webpage can be closed after clicking `Start >`. 

## Protecting Transfers with `tmux`

The `tmux` utility can create persistent shell sessions, and is useful in protecting long file transfers to a remote 
machine against accidental disconnection of your SSH session.

To create a new `tmux` session, use

```shell script
tmux new -s my-session-name
```

To list active `tmux` sessions, use

```shell script
tmux ls
```

To attach to an existing `tmux` session, use

```shell script
tmux attach -t my-session-name-or-id
```

To detach from the current active `tmux` session (while keeping its processes running on the remote machine), use the 
following keyboard shortcut. Note that the `tmux` "prefix" combination (here, `Ctrl-b`) may be different on your system.

```shell script
Ctrl-b + d
```

To kill a `tmux` session and all its processes (so that it does not continue consuming system resources after you're 
finished with it), use

```shell script
tmux kill-session -t my-session-name-or-id
```
