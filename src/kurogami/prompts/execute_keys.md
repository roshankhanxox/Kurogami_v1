It must contain exactly these keys: {keys}. Every one of these keys must appear at
the TOP level of the JSON object -- do not nest them inside a wrapper key or any
other object.
That JSON object is bound to the name `structured` and checked with these exact
Python expressions, so each value must have the type and shape they imply (e.g. a
list where len(...) is compared to a count, a number where it is compared to a
number):
{checks}
Example layout:
```json
{example}
```
