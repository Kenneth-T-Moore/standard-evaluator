# limitations under the License.
""" Utilities for the standard evaluator and its ability to save and load state"""

def unique_names(var):
    names = []
    for my_var in var:
        names.append(my_var.name)
    if len(names) != len(set(names)):
        dup = {x for x in names if names.count(x) > 1}
        raise ValueError(
            f"There is at least one variable / response name that is used multiple times. Names are {dup}"
        )
