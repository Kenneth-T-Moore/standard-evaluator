"""
Created Nov. 29, 2022

@author Mikel Woo
"""
from typing import Optional

from autodocsumm import AutoSummClassDocumenter
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.ext.autodoc import bool_option


class FormatClass(AutoSummClassDocumenter):
    """Adds the :print-options: option to the autoclass directive. Using this
    option will print the unique identifier of the class as well as format the
    available options in an easy to read manner. It can be used as follows:

        .. autoclass:: sample.Class
            :print-options:

    Also adds the :known-solution: option which is used by test evaluators
    to print out the known solution(s). Use the :known-solution: option.

    This feature is implemented similarly to how the autodocsumm extension
    creates the attribute and method summary tables. That extension simply
    overrides the autodoc formatters and adds in the summary table capability.
    To ensure compatability with that extension, this extension does the same
    thing but overriding the autodocsumm class formatter.
    """
    # Set this to have highest priority
    priority = 10 + AutoSummClassDocumenter.priority

    # Copy options
    option_spec = dict(AutoSummClassDocumenter.option_spec)
    # Add print-options and knonw-solution options
    option_spec['print-options'] = bool_option
    option_spec['known-solution'] = bool_option

    def add_content(self, more_content: Optional[StringList]) -> None:
        """This method generates the RST content that will be rendered. When
        Sphinx encounters an autoclass directive it will call this method to
        fill in what content should be in place of the directive.

        Parameters
        ----------
        more_content : Optional[StringList]
            List of strings containing the content of anything placed within the
            directive that is not an option. For example:

                .. autoclass:: sample.Test
                    :print-options:

                    This line will show up
                    As will this one

                    .. math::
                        y = mx + b

                    This line and the equation above will show up

            All the lines (even blank ones) after the blank line below
            :print-options: will be in more_content
        """
        # Call add_content of the autodoc ClassDocumenter which the autosummary
        # AutoSummClassDocumenter derives from. This is to allow us to control
        # when the summary tables are printed
        super(AutoSummClassDocumenter, self).add_content(more_content)

        # Print options if print-options is set and no-print-options isn't
        if 'print-options' in self.options and 'no-print-options' not in self.options:
            self._print_options()

        # Print known solution(s) if known-solution is set and no-known-solution isn't
        if 'known-solution' in self.options and 'no-known-solution' not in self.options:
            self._print_known_solution()

        # Add table of attributes and methods below unique_name and options
        # This method handles the :autosummary: option from autodocsumm. It's
        # placed here to have options printed before the method summary.
        self.add_autosummary(True)

    def _print_options(self):
        """Prints out and formats option information for a given class. If the
        class has a ``unique_name`` attribute then that will be printed as well.
        If the class has an ``options`` attribute with options defined in a
        dictionary, then each option will be printed in a list with its
        explanation, bounds, and default value.
        """
        # Source name tells sphinx which document is being writtne to
        source_name = self.get_sourcename()
        # A reference to the class object
        obj = self.object

        was_printed = False

        # Print unique name
        if hasattr(obj, 'unique_name'):
            # Add space between class description and this
            self.add_line('|', source_name)
            self.add_line('', source_name)
            # Give the title a CSS class for styling
            self.add_line('.. rst-class:: title', source_name)
            self.add_line('', source_name)
            self.add_line(f'**Identifier:** *{obj.unique_name}*', source_name)
            self.add_line('', source_name)

            was_printed = True

        # Print options list
        if hasattr(obj, 'options'):
            self.add_line('|', source_name)
            self.add_line('', source_name)
            self.add_line('.. rst-class:: title', source_name)
            self.add_line('', source_name)
            self.add_line('**Options**', source_name)
            self.add_line('', source_name)

            # Add options-list class to style options list
            self.add_line('.. rst-class:: options-list', source_name)
            self.add_line('', source_name)

            tab = '  '
            # Print option info
            for name, info in obj.options.items():
                # Name
                self.add_line(f'- **{name}**', source_name)
                self.add_line('', source_name)
                # Explanation
                self.add_line('  ' + info['expl'], source_name)
                self.add_line('', source_name)
                # Bounds
                self.add_line(tab + f'- Bounds: {info["bounds"]}', source_name)
                # Default value
                if isinstance(info['value'], (int, float)):
                    self.add_line(tab + f'- Default: {info["value"]:,}', source_name)
                else:
                    self.add_line(tab + f'- Default: {info["value"]}', source_name)

            was_printed = True

        # Add a gap between content printed above and method summary table
        if was_printed:
            self.add_line('', source_name)
            self.add_line('|', source_name)

        # Make sure there is a blank line at the end so nothing breaks
        self.add_line('', source_name)

    def _print_known_solution(self):
        # Source name tells sphinx which document is being writtne to
        source_name = self.get_sourcename()
        # A reference to the class object
        obj = self.object

        # Don't print anything if there is no known_solution property
        if not hasattr(obj, 'known_solution'):
            return

        # Instantiate the class so we can access it's properties
        # Skip classes that aren't fully defined (ie. missing abstract methods)
        try:
            inst = obj()
        except TypeError:
            return

        # Get known solution, variable names, and response names
        sol = inst.known_solution
        variables = list(inst.problem['variables'].keys())
        responses = list(inst.problem['responses'].keys())

        # Add space between docstring and solution table
        self.add_line('|', source_name)
        self.add_line('', source_name)
        # Add CSS title class to format Known Solution title
        self.add_line('.. rst-class:: title', source_name)
        self.add_line('', source_name)

        # No known solution
        if sol is None:
            self.add_line('**Known Solution**', source_name)
            self.add_line('', source_name)
            self.add_line('No known solution!', source_name)
        # Print known solution(s)
        else:
            # Print title
            if len(sol) == 1:
                self.add_line('**Known Solution**', source_name)
            else:
                self.add_line('**Known Solutions**', source_name)

            self.add_line('', source_name)

            tab = ' '*4

            # Create table with a row for each solution
            self.add_line('.. list-table::', source_name)
            # Table will span whole width of rendered area
            self.add_line(tab + ':width: 100', source_name)
            # Only one header row
            self.add_line(tab + ':header-rows: 1', source_name)
            self.add_line('', source_name)

            # Fill in header row
            for i, name in enumerate(variables + responses):
                if i == 0:
                    self.add_line(tab + f'* - {name}', source_name)
                else:
                    self.add_line(tab + f'  - {name}', source_name)

            # Fill in table
            for i in range(len(sol)):
                row = sol.iloc[i]

                for j, name in enumerate(variables + responses):
                    if j == 0:
                        self.add_line(tab + f'* - {row[name]}', source_name)
                    else:
                        self.add_line(tab + f'  - {row[name]}', source_name)

        # Add gap between this and content below
        self.add_line('', source_name)
        self.add_line('|', source_name)
        self.add_line('', source_name)


def setup(app: Sphinx) -> None:
    """Loads in required extensions and registers actions, documenters, etc.
    with Sphinx to allow them to be used.

    Parameters
    ----------
    app : Sphinx
        The current Sphinx instance.
    """
    # Load the autodoc and autodocsumm extentsions since both are required
    app.setup_extension('sphinx.ext.autodoc')
    app.setup_extension('autodocsumm')

    # Register this custom documenter with Sphinx
    # Set override to True so that warnings are not given about existing name
    app.add_autodocumenter(FormatClass, override=True)
