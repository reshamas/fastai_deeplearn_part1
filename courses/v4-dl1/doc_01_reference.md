- `?` gives interactive python guide 

```text
IPython -- An enhanced Interactive Python
=========================================

IPython offers a fully compatible replacement for the standard Python
interpreter, with convenient shell features, special commands, command
history mechanism and output results caching.

At your system command line, type 'ipython -h' to see the command line
options available. This document only describes interactive features.

GETTING HELP
------------

Within IPython you have various way to access help:

  ?         -> Introduction and overview of IPython's features (this screen).
  object?   -> Details about 'object'.
  object??  -> More detailed, verbose information about 'object'.
  %quickref -> Quick reference of all IPython specific syntax and magics.
  help      -> Access Python's own help system.

If you are in terminal IPython you can quit this screen by pressing `q`.


MAIN FEATURES
-------------

* Access to the standard Python help with object docstrings and the Python
  manuals. Simply type 'help' (no quotes) to invoke it.

* Magic commands: type %magic for information on the magic subsystem.

* System command aliases, via the %alias command or the configuration file(s).

* Dynamic object information:

  Typing ?word or word? prints detailed information about an object. Certain
  long strings (code, etc.) get snipped in the center for brevity.

  Typing ??word or word?? gives access to the full information without
  snipping long strings. Strings that are longer than the screen are printed
  through the less pager.

  The ?/?? system gives access to the full source code for any object (if
  available), shows function prototypes and other useful information.

  If you just want to see an object's docstring, type '%pdoc object' (without
  quotes, and without % if you have automagic on).

* Tab completion in the local namespace:

  At any time, hitting tab will complete any available python commands or
  variable names, and show you a list of the possible completions if there's
  no unambiguous one. It will also complete filenames in the current directory.

* Search previous command history in multiple ways:

  - Start typing, and then use arrow keys up/down or (Ctrl-p/Ctrl-n) to search
    through the history items that match what you've typed so far.

  - Hit Ctrl-r: opens a search prompt. Begin typing and the system searches
    your history for lines that match what you've typed so far, completing as
    much as it can.

  - %hist: search history by index.

* Persistent command history across sessions.

* Logging of input with the ability to save and restore a working session.

* System shell with !. Typing !ls will run 'ls' in the current directory.

* The reload command does a 'deep' reload of a module: changes made to the
  module since you imported will actually be available without having to exit.

* Verbose and colored exception traceback printouts. See the magic xmode and
  xcolor functions for details (just type %magic).

* Input caching system:

  IPython offers numbered prompts (In/Out) with input and output caching. All
  input is saved and can be retrieved as variables (besides the usual arrow
  key recall).

  The following GLOBAL variables always exist (so don't overwrite them!):
  _i: stores previous input.
  _ii: next previous.
  _iii: next-next previous.
  _ih : a list of all input _ih[n] is the input from line n.

  Additionally, global variables named _i<n> are dynamically created (<n>
  being the prompt counter), such that _i<n> == _ih[<n>]

  For example, what you typed at prompt 14 is available as _i14 and _ih[14].

  You can create macros which contain multiple input lines from this history,
  for later re-execution, with the %macro function.

  The history function %hist allows you to see any part of your input history
  by printing a range of the _i variables. Note that inputs which contain
  magic functions (%) appear in the history with a prepended comment. This is
  because they aren't really valid Python code, so you can't exec them.

* Output caching system:

  For output that is returned from actions, a system similar to the input
  cache exists but using _ instead of _i. Only actions that produce a result
  (NOT assignments, for example) are cached. If you are familiar with
  Mathematica, IPython's _ variables behave exactly like Mathematica's %
  variables.

  The following GLOBAL variables always exist (so don't overwrite them!):
  _ (one underscore): previous output.
  __ (two underscores): next previous.
  ___ (three underscores): next-next previous.

  Global variables named _<n> are dynamically created (<n> being the prompt
  counter), such that the result of output <n> is always available as _<n>.

  Finally, a global dictionary named _oh exists with entries for all lines
  which generated output.

* Directory history:

  Your history of visited directories is kept in the global list _dh, and the
  magic %cd command can be used to go to any entry in that list.

* Auto-parentheses and auto-quotes (adapted from Nathan Gray's LazyPython)

  1. Auto-parentheses
        
     Callable objects (i.e. functions, methods, etc) can be invoked like
     this (notice the commas between the arguments)::
       
         In [1]: callable_ob arg1, arg2, arg3
       
     and the input will be translated to this::
       
         callable_ob(arg1, arg2, arg3)
       
     This feature is off by default (in rare cases it can produce
     undesirable side-effects), but you can activate it at the command-line
     by starting IPython with `--autocall 1`, set it permanently in your
     configuration file, or turn on at runtime with `%autocall 1`.

     You can force auto-parentheses by using '/' as the first character
     of a line.  For example::
       
          In [1]: /globals             # becomes 'globals()'
       
     Note that the '/' MUST be the first character on the line!  This
     won't work::
       
          In [2]: print /globals    # syntax error

     In most cases the automatic algorithm should work, so you should
     rarely need to explicitly invoke /. One notable exception is if you
     are trying to call a function with a list of tuples as arguments (the
     parenthesis will confuse IPython)::
       
          In [1]: zip (1,2,3),(4,5,6)  # won't work
       
     but this will work::
       
          In [2]: /zip (1,2,3),(4,5,6)
          ------> zip ((1,2,3),(4,5,6))
          Out[2]= [(1, 4), (2, 5), (3, 6)]

     IPython tells you that it has altered your command line by
     displaying the new command line preceded by -->.  e.g.::
       
          In [18]: callable list
          -------> callable (list)

  2. Auto-Quoting
    
     You can force auto-quoting of a function's arguments by using ',' as
     the first character of a line.  For example::
       
          In [1]: ,my_function /home/me   # becomes my_function("/home/me")

     If you use ';' instead, the whole argument is quoted as a single
     string (while ',' splits on whitespace)::
       
          In [2]: ,my_function a b c   # becomes my_function("a","b","c")
          In [3]: ;my_function a b c   # becomes my_function("a b c")

     Note that the ',' MUST be the first character on the line!  This
     won't work::
       
          In [4]: x = ,my_function /home/me    # syntax error
          
```
