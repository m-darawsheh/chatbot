
# what is virtual environments and why do we need it ?

A virtual environment is like a separate folder where Python installs packages just for your project,
instead of installing them globally on your system.
This keeps your projects organized and avoids conflicts between package versions.

# how to create a virtual enviroment ?

python -m venv venv

# notes

* you need to activate it by this command (source venv/bin/activate)
* know what ever lib you install will be only this project
* to deactivate it type in the tirminal(deactivate)

* now we both will install packages, python has provide a good way to tell me what packages i should install
to let the program work with me.
* now lets say you just installed (flask, torch) inisted of telling me to install them you just type this command
(pip freeze > requirements.txt) (pip freeze) will show all the packages and (> requirements) [minishell]
* now when i get pull and will find requirements.txt file then will just type pip install -r requirements.txt
and all the packages will be installed.
