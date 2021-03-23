## Here is a 10 second snapshot of training on all 32 levels in parallel.
![training](https://github.com/jmphil09/mario_rl/blob/master/training_gif1.gif)
![training](https://github.com/jmphil09/mario_rl/blob/master/training_gif2.gif)

# Current Task
Create a new GameRunner class to run multiple levels and sync the results in one main model. Example, instead of training on one level, train on all 32 levels (in parallel!) and then update the model.

# Other Tasks
- Replace neat-python library (possibly build a custom library for fun)
- Build a custom checkpoint system (or find out how to work around neat-python index bug)
- Add info on how the "good" config files were generated - the config with timestamp 1566017738 is one of the "best"
- Plot random hyperparam results as a function of population size. See if there are any small population sizes in the top 20 or so. Add this to README.
- Move generated data to external storage with a README on what each data folder is, and whether or not an experiment has been run on it. (The data not included in this repo is many GB, but some should be kept for further research.)
- Finish the main README file :)
- Add documentation on hyperparam searching with some data and graphs.

# Helpful Commands
To see the list of all roms supported (and their name in gym-retro)
- `import retro`
- `print(retro.data.list_games())`

To import roms, go to the location of the rom and run:
- `python -m retro.import .`

# Side project: Apply Neural Style Transfer (NST) to a result
Save a result as a gif or video and apply NST to the result.

Some ideas for styles: Art like cubism and starry night, Animated Cartoons like The Simpsons and Family Guy, other games like Sonic, Mario RPG / Donkey Kong Country, Atari games

## NST tasks
- Test different NST libraries like the popular tensorflow and pytorch libraries
- Apply NST to static image of mario result
- Apply NST to gif/video
- Try multiple NST styles
