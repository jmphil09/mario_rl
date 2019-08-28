# Setup
- Documentation coming soon

# Examples
- Documentation coming soon

# Current Tasks
1) Plot random hyperparam results as a function of population size. See if there are any small population sizes in the top 20 or so.
2) Move generated data to external storage with a README on what each data folder is, and whether or not an experiment has been run on it. (The data not included in this repo is many GB, but some should be kept for further research.)
3) Finish the main README file :)
4) Add documentation on hyperparam searching with some data and graphs.


# Helpful Commands
To see the list of all roms supported (and their name in gym-retro)
- `import retro`
- `print(retro.data.list_games())`

To import roms, go to the location of the rom and run:
- `python -m retro.import .`
