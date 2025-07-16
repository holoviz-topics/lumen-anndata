import panel as pn
import panel_material_ui as pmui
import param

from panel.custom import PyComponent

pn.extension()


class AutoCompleteMultiChoice(PyComponent):
    """
    A composite component combining a text input with a MultiChoice widget.
    The text input serves as a key for a dictionary where each key maps to a list of selected values.
    """

    value = param.Dict(default={}, doc="Dictionary mapping keys to lists of selected values")

    options = param.List(default=[], doc="List of available options for the MultiChoice")

    placeholder = param.String(default="Enter key name", doc="Placeholder text for the input")

    _input_key = param.String(default="", doc="Current value of the text input (key)")

    _input_value = param.String(default="", doc="Current value of the text input (value)")

    _current_key = param.String(default="", doc="Currently selected key")

    _current_selection = param.List(default=[], doc="Current selection for the active key")

    def __init__(self, **params):
        super().__init__(**params)

        self._key_input = pmui.AutocompleteInput.from_param(
            self.param._input_key, placeholder=self.param.placeholder, name="Group Key", description=None, restrict=False
        )

        self._value_input = pmui.AutocompleteInput.from_param(
            self.param._input_value,
            options=self.param.options,
            placeholder="Enter value",
            name="Value",
            description=None,
            restrict=False,
            disabled=self.param._input_key.rx().rx.len() == 0,
        )

        self._multi_choice = pmui.MultiChoice.from_param(
            self.param._current_selection,
            options=self.param.options,
            name="Grouped Selection",
            description=None,
            disabled=self._value_input.param.disabled,
        )

        self._json_editor = pn.widgets.JSONEditor.from_param(
            self.param.value,
            name="JSON Editor",
            mode="tree",
            width=300,
            height=400,
        )

    @param.depends("_input_key", watch=True)
    def _handle_key_input(self):
        """Handle when a key is entered in the text input"""
        key = self._input_key.strip()
        if key:
            # Set the current key
            self._current_key = key

            # Initialize the key in the value dict if it doesn't exist
            if key not in self.value:
                new_value = dict(self.value)
                new_value[key] = []
                self.value = new_value

            # Update current selection to match the key's current values
            self._current_selection = list(self.value.get(key, []))

            if key not in self._key_input.options:
                self._key_input.options = self._key_input.options + [key]

    @param.depends("_input_value", watch=True)
    def _handle_value_input(self):
        """Handle when a value is entered in the text input"""
        value = self._input_value.strip()
        if value and self._current_key:
            if value not in self.options:
                self.options = self.options + [value]

            # Add the value to the current selection for the active key
            if value not in self._current_selection:
                self._current_selection = self._current_selection + [value]
                self.value[self._current_key] = self._current_selection
            self._value_input.value = ""

    @param.depends("_current_selection", watch=True)
    def _handle_selection_change(self):
        """Handle when the MultiChoice selection changes"""
        if self._current_key:
            # Update the value dict with the new selection
            new_value = dict(self.value)
            new_value[self._current_key] = list(self._current_selection)
            self.value = new_value

    def __panel__(self):
        return pn.Column(self._key_input, self._value_input, self._multi_choice, self._json_editor, sizing_mode="stretch_both")
