It must contain exactly these keys: {keys}. Every one of these keys must appear at
the TOP level of the JSON object -- do not nest them inside a wrapper key or any
other object.
That JSON object is bound to the name `structured` and checked with these exact
Python expressions, so each value must have the type and shape they imply (e.g. a
list where len(...) is compared to a count, a number where it is compared to a
number):
{checks}
These checks are binding: your answer fails if any of them is False. If you believe
one of them is wrong for this situation, still meet it, and say in your prose why
you disagree -- a reviewer reads that.
Example layout:
```json
{example}
```
