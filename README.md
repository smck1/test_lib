# test_lib
Creating my first library

# Installation
```
pip install my_lib
```

# Example use-case
```
python run.py
```

# Guides used
https://redandgreen.co.uk/sphinx/ 

```console
(my_lib) C:\Users\aabywan\Downloads\test_lib>sphinx-apidoc -o ./docs ./phaser
```

https://redandgreen.co.uk/sphinx-to-github-pages-via-github-actions/
https://towardsdatascience.com/deep-dive-create-and-publish-your-first-python-library-f7f618719e14


# Configuring Sphinx to autodoc subfolders!
- In the root folder of the git repository, create a "docs" folder
    - ```mkdir docs```
- Change directory to the docs folder
    - ```cd docs```
- Run the quickstart guide 
    - ```sphinx-quickstart```
    - Press "n" when asked to split folders
- Then use sphinx-apidoc to automatically create the .rst files
    - ```sphinx-apidoc -o ../docs ../phaser```
- Modify the "./docs/conf.py" file and add ... 
    ```python
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    #
    extensions = ["sphinx.ext.autodoc"]
    #
    html_theme = 'sphinx_rtd_theme'
    ```
- Add 'modules.rst' and other .rst files to 'index.rst'
    ```rst
    .. toctree::
    :maxdepth: 2
    :caption: Contents:

    example
    modules
    ```
- From the docs directory, make the html code
    ```
    make html
    ```