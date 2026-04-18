---
owning-stage: "~devops::verify"
title: "The CI Steps Expression Language"
toc_hide: false
---

## Summary

This document proposes extending the CI Steps Expression Language to include string manipulation, arithmetic, comparisons,
logic, property access, and function calls. It proposes to build the language in-house, and explains why it differs from the CI Components Expression Language.

The document then formally specifies the CI Steps Expression language.

## Terms

- **CI Steps Expression Language**: The syntax used by Step authors and users to reference variables within Steps. These expressions use `${{ }}` delimiters. Examples include:
  - Accessing an input: `${{ inputs.address }}`
  - Referencing a job variable: `${{ job.GITLAB_USER_NAME }}`
  - Getting a previous step's output: `${{ step.my_step.output.duration }}`
- **CI Components Expression Language**: The syntax used to reference CI Pipeline and CI Component inputs. These expressions use `$[[ ]]` delimiters. For example, you can create a component with a dynamically named job: `$[[ inputs.job-prefix ]]-scan-website:`
- **Evaluation**: The process of resolving an expression to its actual value. For instance, the expression `${{ inputs.speed_of_light_ms }}` evaluates to `299792458` when that value is provided in the inputs.

## Motivation

A differentiator of CI Steps compared with other CI solutions is that individual job tasks (steps) may be reused. Users create a Steps CI job by stitching together purpose-built
steps when performing a custom task (e.g. email my stakeholders), and steps from an external source when performing a common task (e.g. a compile Go step hosted in a steps library).

Steps are stitched together by using expressions `${{ [expression] }}` as the step input. The expression is evaluated during job execution, so has access to job variables or outputs of a previously run step.
For example, if a Docker `build` step builds a Docker image, a subsequent `release` step can use the image reference output from the build step to ensure the correct image is released.

```yaml
build-and-release:
  run:
    - name: build
      step: steps/library/docker/build # this step has an output called 'image_ref', the repository and tag of the Docker image
      inputs:
        work_dir: .
        Dockerfile: ./Dockerfile
    - name: release
      step: steps/library/docker/promote
      inputs:
        from_image: ${{ steps.build.outputs.image_ref }} # expression evaluates to the name of the image created in the 'build' step
        to_image: registry.gitlab.com/my-product:1.0.1
```

The above example is simple, the `image_ref` output is passed directly into the `from_image` input.
Many real-world situations are more complex, so the user needs tools to craft input values from variables and step outputs. Examples include:

- String manipulation: `build_image: ${{ job.CI_REGISTRY }}/my-build:${{ job.CI_PIPELINE_IID }}`
- Arithmetic: `version: ${{ steps.versions.most_recent.major + 1 }}`
- Comparisons: `log_when: ${{ inputs.LOGLEVEL > 4 }}`
- Logic: `user: ${{ inputs.username || "default_username" }}`
- Property access: `address_line2: ${{ user.address.street[2] }}`
- Function calls: `commit: ${{ replace(job.CI_COMMIT_DESCRIPTION, "\n", " ") }}`

If these features are not added to the expression language, then the burden of this variable/step output to step input conversion is passed on to the step author. This:

- Reduces the productivity of the step author, and
- Reduces the reusability of the step (if you don't have the right input format, it can't be used)

## Goal

### Summary

Extend the CI Steps Expression Language to allow string manipulation, arithmetic, comparisons, logic, property access, and function calls.

The extension to the expression language:

- MUST be backwards compatible with already used property lookups, e.g. `${{ inputs.my_variable }}` so it can replace the current implementation
- MUST be specified as a context-free grammar using Extended Backus–Naur form (EBNF) so the language can be reviewed before it is implemented
- MUST be extensible. Additional functions, operations, and mutations are possible future extensions
- SHOULD be easy to use, and feel somewhat familiar to engineers. More powerful than JSON, less complex than JavaScript

### Built in-house

Expressions will be built in-house at GitLab using a recursive-descent parser.

A review of existing expression languages and libraries didn't find anything suitable for CI Steps' needs. The main issues were:

- They offer too many features, are too complex, or are considered to be unfamiliar to GitLab users
- They do not support passing metadata, so Steps can't determine if evaluated expressions are derived from sensitive values

### Out-of-scope

This proposal does not:

- Outline which functions should be defined, only that functions can be called
- Outline which namespaces/scopes/contexts/variables should be defined, only that they can be used
- Change the CI Components Expression Language
- Propose ways to implement control flow

## Example use-cases

### Version management

```yaml
# Extract and increment semantic versions
- name: bump-major-version
  step: ./steps/bump
  inputs:
    new_version: ${{ major_version(steps.get_current.outputs.version) + 1 }}.0.0
    tag_name: v${{ steps.get_current.outputs.version }}
```

### Environment-Specific Configuration

```yaml
# Different registry URLs based on branch
- name: deploy
  step: ./steps/deploy
  inputs:
    registry: '${{ (job.CI_COMMIT_REF_NAME == "main" && "prod.registry.com") || "staging.registry.com" }}'
    replicas: '${{ (job.CI_COMMIT_REF_NAME == "main" && 5) || 2 }}'
```

### Conditional Execution Logic

```yaml
# Skip steps based on file changes or conditions
- name: run-tests
  step: ./steps/run-tests
  inputs:
    skip_integration: '${{ !contains(steps.changes.outputs.files, "integration/") }}'
    test_command: '${{ (inputs.test_type == "full" && "npm run test:all") || "npm run test:unit" }}'
```

### Security & Compliance

```yaml
# Validate and transform security scan results
- name: security_gate
  step: ./security_gate
  inputs:
    proceed: ${{ steps.scan.outputs.critical_vulnerabilities == 0 && steps.scan.outputs.high_vulnerabilities < 5 }}
```

## Deviation from CI Components

### As is - CI Components

The CI Components Expression Language:

- Is expressed surrounded by `$[[` and `]]`
- Is evaluated during pipeline creation
- Supports property access using `.`, for example, `inputs.rust_version`
- Supports types `array`, `boolean`, `number` and `string`
- Supports string templating, for example, `echo $[[inputs.message]]`
- Supports function calls with pipes `|`, for example, `$[[ inputs.test | expand_vars | truncate(5,8) ]]`
  - Only three piped functions are supported
  - Only supports predefined functions, of which there are two, `expand_vars` and `truncate`

### To be - CI Steps

The CI Steps Expression Language MUST:

- Support types `array`, `boolean`, `number`, `string` AND `struct`
- Be evaluated at the last possible moment, during job execution
- Support powerful ways to manipulate variables/job inputs/step outputs into step inputs, using arithmetic, string manipulation, logic, and property access
- Support comparisons for control flow
- Support complex function composition, for example, `max(15, major_version(extract_version("postgres:13.4.1")))`
- Support string templating
- Support property access using `.`

### Moving forward

The Steps and CI Components expression languages are different because they are evaluated at different times using different contexts.

- Steps cannot use CI Components expressions, it does not provide rich enough tools for the user to craft step input values.
- CI Components cannot use Steps expressions, step context such as `env`, `output_file`, `work_dir`, and `export_file` are not available during pipeline creation.

While differences between the expression languages remain, effort should be made to minimize differences where possible. Going forward, Steps expressions will:

- Be surrounded by `${{` and `}}` to communicate to a user both the expression language used, and that evaluation happens during job execution
- Conform to CI Components expressions where possible, for example:
  - Property access
  - Types `array`, `boolean`, `number` and `string`
- Deviate from CI Components expressions where necessary, for example:
  - Support `struct`
  - Support arithmetic, string manipulation, logic, comparisons
  - The way functions are called, limits on number of functions used, the functions available to call

See https://gitlab.com/groups/gitlab-org/-/epics/18519+ to follow the effort for unifying the CI Component and Steps expression languages.

### Example

The following example is of a CI component containing jobs that run steps. Users are required to know two expression syntaxes.

```yaml
spec:
  inputs:
    echo_version:
      type: string
---

build-job:
  run:
    - name: echo_step
      step: gitlab.com/steps/echo@$[[inputs.echo_version]]  # CI Component expression, evaluated when the pipeline is created
      inputs:
        message: 'Hello, ${{ remove_new_lines(jobs.CI_RUNNER_DESCRIPTION) }}' # CI Steps expression, evaluated during job execution
```

## Specification

### Context-free grammar

The CI Steps Expression Language defined as an Extended Backus-Naur Form (EBNF) grammar.

```ebnf
// Lexical elements
unicode_char   = /* an arbitrary Unicode code point */ .
unicode_letter = /* a Unicode code point categorized as "Letter" */ .
unicode_digit  = /* a Unicode code point categorized as "Number, decimal digit" */ .

letter = unicode_letter | "_" .
digit  = "0" … "9" .

// String escape sequences
escaped_single  = `\` ( `\` | "'" ) .
escaped_double  = `\` ( "a" | "b" | "f" | "n" | "r" | "t" | "v" | `\` | "/" | `"` | "$" ) .
escaped_unicode = "u" digit digit digit digit .

// Template expressions
template = "${{" Expression "}}" .

// Tokens (lexical rules)
identifier = letter { letter | unicode_digit } . /* except reserved */
int_lit    = digit { digit } .
float_lit  = digit { digit } "." digit { digit } [ exponent ] .
exponent   = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
number     = int_lit | float_lit .

string_lit     = single_quoted | double_quoted .
single_quoted  = "'" { unicode_char | escaped_single } "'" .
double_quoted  = `"` { unicode_char | escaped_double | escaped_unicode | template } `"` .
string         = string_lit .

// Operators
binary_op = "||" | "&&" | rel_op | add_op | mul_op .
unary_op  = "+" | "-" | "!" .
rel_op    = "==" | "!=" | "<" | "<=" | ">" | ">=" .
add_op    = "+" | "-" .
mul_op    = "*" | "/" | "%" .

// Reserved words
reserved = "array" | "as" | "break" | "case" | "const" | "continue" |
    "default" | "else" | "fallthrough" | "float" | "for" | "func" |
    "function" | "goto" | "if" | "import" | "in" | "int" | "let" | "loop" |
    "map" | "namespace" | "number" | "object" | "package" | "range" |
    "return" | "string" | "struct" | "switch" | "type" | "var" | "void" |
    "while" .

// CI Steps Expression Language
Expression = OrExpression .

OrExpression = AndExpression { "||" AndExpression } .
AndExpression = ComparisonExpression { "&&" ComparisonExpression } .
ComparisonExpression = AdditiveExpression { rel_op AdditiveExpression } .
AdditiveExpression = MultiplicativeExpression { add_op MultiplicativeExpression } .
MultiplicativeExpression = UnaryExpression { mul_op UnaryExpression } .

UnaryExpression = unary_op UnaryExpression
                | PostfixExpression .

PostfixExpression = PrimaryExpression
                  { "." identifier
                  | "[" Expression "]"
                  | Call
                  } .

PrimaryExpression = Literal
                  | identifier
                  | "(" Expression ")"
                  | Array
                  | Object .

Literal = "null"
        | "true"
        | "false"
        | string
        | number .

Array = "[" [ ArrayElements ] "]" .
ArrayElements = Expression { "," Expression } [ "," ] .

Object = "{" [ ObjectElements ] "}" .
ObjectElements = ObjectElement { "," ObjectElement } [ "," ] .
ObjectElement = Expression ":" Expression .

Call = "(" [ Expression { "," Expression } ] ")" .
```

### Draft implementation

- [Expression playground](https://storage.googleapis.com/directory-io/playground/index.html)
- [Draft: Add specification](https://gitlab.com/gitlab-org/step-runner/-/merge_requests/239)
- [Draft: Implement expression lexer](https://gitlab.com/gitlab-org/step-runner/-/merge_requests/234)
- [Draft: Implement expression value/type system](https://gitlab.com/gitlab-org/step-runner/-/merge_requests/242)
- [Draft: Implement expression parser](https://gitlab.com/gitlab-org/step-runner/-/merge_requests/243)
- [Draft: Implement expression evaluator](https://gitlab.com/gitlab-org/step-runner/-/merge_requests/244)

### Breakdown and examples

#### Source code representation

Source code is Unicode text encoded in UTF-8. The text is not canonicalized, so a single accented code point is distinct from the same character constructed from combining an accent and a letter.

##### Characters

```ebnf
unicode_char   = /* an arbitrary Unicode code point */ .
unicode_letter = /* a Unicode code point categorized as "Letter" */ .
unicode_digit  = /* a Unicode code point categorized as "Number, decimal digit" */ .
```

##### Letters and digits

```ebnf
letter = unicode_letter | "_" .
digit  = "0" … "9" .
```

#### Lexical elements

##### Comments

The language does not currently support comments.

##### Tokens

Tokens form the vocabulary of the language. There are four classes: identifiers, keywords, operators and punctuation, and literals.

##### Identifiers

Identifiers name variables and functions.

```ebnf
identifier = letter { letter | unicode_digit } .
```

Identifiers must not be keywords. Identifiers are case-sensitive: `foo`, `Foo`, and `FOO` are three different identifiers.

##### Keywords

The following keywords are reserved and may not be used as identifiers:

```text
array       as          break       case        const
continue    default     else        fallthrough float
for         func        function    goto        if
import      in          int         let         loop
map         namespace   number      object      package
range       return      string      struct      switch
type        var         void        while
```

Additionally, the following literal keywords are recognized:

```text
false       null        true
```

##### Operators and punctuation

The following character sequences represent operators and punctuation:

```text
+    &&    ==    !=    (    )
-    ||    <     <=    [    ]
*    !     >     >=    {    }
/    %     .     ,     :
```

##### Integer literals

Integer literals are sequences of digits. Leading zeros are permitted.

```ebnf
int_lit = digit { digit } .
```

##### Floating-point literals

Floating-point literals consist of an integer part, a decimal point, a fractional part, and an optional exponent part.

```ebnf
float_lit = digit { digit } "." digit { digit } [ exponent ] .
exponent  = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
```

##### Number literals

A number literal is either an integer or a floating-point literal.

```ebnf
number = int_lit | float_lit .
```

##### String literals

String literals represent strings constants and come in two forms: single-quoted and double-quoted. Both allow multiple characters, but differ in how they handle escape sequences and template expressions.

Single-quoted strings are raw string literals. Use single quotes when you want a string with minimal interpretation.

- They do not support template expressions or most escape sequences
- Supports minimal escape sequences
  - `\\` - backslash
  - `\'` - single quote

Double-quoted strings support:

- Template expressions using `${{ ... }}` syntax. The expression inside a template must evaluate to a string. See [Template Expressions](#template-expressions) for details
- A full set of escape sequences
  - `\a` - alert or bell
  - `\b` - backspace
  - `\f` - form feed
  - `\n` - newline
  - `\r` - carriage return
  - `\t` - horizontal tab
  - `\v` - vertical tab
  - `\\` - backslash
  - `\/` - forward slash
  - `\"` - double quote
  - `\$` - single dollar sign

```ebnf
string_lit         = single_quoted | double_quoted .
single_quoted      = "'" { unicode_char | escaped_single } "'" .
double_quoted      = `"` { unicode_char | escaped_double | escaped_unicode | template } `"` .
escaped_single     = `\` ( `\` | "'" ) .
escaped_double     = `\` ( "a" | "b" | "f" | "n" | "r" | "t" | "v" | `\` | "/" | `"` | "$" ) .
escaped_unicode    = "u" digit digit digit digit .
template           = "${{" Expression "}}" .
```

Examples:

```js
// Single-quoted strings
'Hello, world!'
'It\'s a beautiful day'  // Escaped single quote
'Path: C:\\Users\\Alice' // Escaped backslashes
'${{ "hello" }}'         // Treated literally, does not evaluate

// Double-quoted strings
"Hello, world!"
"She said, \"Hello!\""   // Escaped double quotes
"Line 1\nLine 2\nLine 3" // Newline characters
"Name:\tJohn\nAge:\t30"  // Tab and newline
"Alert\a\tBackspace\b"   // Special characters

// Template expressions
"Hello, ${{ name }}!"                                 // Simple variable interpolation
"Path: ${{ dir }}/${{ file }}"                       // Multiple templates
```

#### Types

The language supports the following types:

##### Boolean

Boolean values are represented by the predeclared constants `true` and `false`.

##### Null

The null value is represented by the predeclared constant `null`.

##### Number

Numbers are IEEE 754 double-precision floating-point values (53 bits of precision).

##### String

Strings are immutable sequences of Unicode code points.

##### Array

Arrays are ordered sequences of values. Elements can be of any type and types can be mixed within an array.

##### Object

Objects are unordered collections of key-value pairs. Keys must be strings (either string literals or expressions that evaluate to strings). Values can be of any type.

#### Type Operations

This section describes which operations are valid between different types and their behavior.

##### Type compatibility table

| Operation    | Valid Types     | Result Type | Notes                                |
|--------------|-----------------|-------------|--------------------------------------|
| `+` (binary) | number + number | number      | Addition                             |
| `+` (binary) | string + string | string      | Concatenation                        |
| `-` (binary) | number - number | number      | Subtraction                          |
| `*`          | number * number | number      | Multiplication                       |
| `/`          | number / number | number      | Division (error on divide by zero)   |
| `%`          | number % number | number      | Modulo (error on modulo by zero)     |
| `+` (unary)  | number          | number      | Identity (returns unchanged)         |
| `-` (unary)  | number          | number      | Negation                             |
| `!`          | any             | boolean     | Logical NOT based on truthiness      |
| `==`         | any == any      | boolean     | Equality comparison                  |
| `!=`         | any != any      | boolean     | Inequality comparison                |
| `<`          | any < any       | boolean     | Less than (see comparison semantics) |
| `<=`         | any <= any      | boolean     | Less than or equal                   |
| `>`          | any > any       | boolean     | Greater than                         |
| `>=`         | any >= any      | boolean     | Greater than or equal                |
| `&&`         | any && any      | any         | Returns first falsy or last value    |
| `\|\|`       | any \|\| any    | any         | Returns first truthy or last value   |
| `.`          | object/array    | any         | Property/method access               |
| `[]`         | object[string]  | any         | Object property access               |
| `[]`         | array[number]   | any         | Array element access                 |
| `()`         | function        | any         | Function call                        |

##### Equality semantics

The `==` and `!=` operators compare values as follows:

- **null**: Only equals `null`
- **boolean**: Only equals boolean with same value
- **number**: Equals numbers with same numeric value
- **string**: Equals strings with identical UTF-8 byte sequences
- **array**: Equals arrays with same length and equal elements (deep comparison)
- **object**: Equals objects with same keys and equal values (deep comparison, key order irrelevant)

##### Comparison semantics

The comparison operators `<`, `<=`, `>`, and `>=` can compare any types. When comparing values:

1. **Same type comparisons**:
    - **numbers**: Numeric comparison
    - **strings**: Lexicographic comparison (UTF-8 byte order)
    - **booleans**: `false < true`
    - **arrays**: Compares elements in order; an array with fewer items is less than an array with more items
    - **objects**: Compares elements in sorted key order; an object with fewer items is less than an object with more items
    - **null**: All null values are equal

2. **Different type comparisons**:

   When comparing values of different types, types are ordered as follows:

   - `null`
   - `boolean`
   - `number`
   - `string`
   - `array`
   - `object`

   For example:

   ```js
   null < true      // true (null has order 0, boolean has order 1)
   42 < "hello"     // true (number has order 2, string has order 3)
   [1,2,3] > "text" // true (array has order 4, string has order 3)
   ```

##### Short-circuit evaluation

The logical operators `&&` and `||` use short-circuit evaluation:

- `&&`: If left operand is falsy, right operand is not evaluated
- `||`: If left operand is truthy, right operand is not evaluated

#### Expressions

##### Primary expressions

Primary expressions are the operands for unary and binary expressions.

```ebnf
PrimaryExpression = Literal | identifier | "(" Expression ")" | Array | Object .
Literal           = "null" | "true" | "false" | string_lit | number .
```

Parentheses can be used to group expressions and override operator precedence:

```js
2 + 3 * 4        // evaluates to 14
(2 + 3) * 4      // evaluates to 20
```

##### Array literals

Array literals construct array values.

```ebnf
Array         = "[" [ ArrayElements ] "]" .
ArrayElements = Expression { "," Expression } [ "," ] .
```

Example:

```js
[1, 2, 3]
["a", 1, true, null]
[1, 2, 3,]  // trailing comma allowed
```

##### Object literals

Object literals construct object values. Keys can be string literals or expressions that evaluate to strings.

```ebnf
Object         = "{" [ ObjectElements ] "}" .
ObjectElements = ObjectElement { "," ObjectElement } [ "," ] .
ObjectElement  = Expression ":" Expression .
```

Example:

```js
{"name": "John", "age": 30}
{"key": value}
{computed_key: value}          // computed_key must evaluate to string
{"prefix" + "_suffix": value}  // expressions that produce strings
{obj.prop: value,}             // trailing comma allowed
```

Note: Object keys must evaluate to strings at runtime. Non-string keys will result in a runtime error.

##### Selectors

Selectors access properties or elements of a value.

```ebnf
Selector = "." identifier | "[" Expression "]" .
```

Property access with `.` requires an identifier. Computed property access with `[]` accepts any expression.

Example:

```js
obj.property
obj["property"]
my_array[0]
my_array[index]
```

##### Function calls

Function calls invoke a function with zero or more arguments.

```ebnf
Call = "(" [ Expression { "," Expression } ] ")" .
```

Example:

```js
my_func()
my_func(1, 2, 3)
obj.method()
my_array[0]()  // if my_array[0] contains a function
```

##### Unary operators

Unary operators have the highest precedence.

```ebnf
UnaryExpression = unary_op UnaryExpression | PostfixExpression .
unary_op        = "+" | "-" | "!" .
```

| Operator | Name | Types | Description |
|----------|------|-------|-------------|
| `+` | unary plus | number | numeric identity |
| `-` | unary minus | number | numeric negation |
| `!` | logical NOT | any | logical negation (based on truthiness) |

##### Binary operators

Binary operators are left-associative and follow standard precedence rules.

| Precedence | Operators | Associativity |
|------------|-----------|---------------|
| 5 | `*` `/` `%` | left |
| 4 | `+` `-` | left |
| 3 | `==` `!=` `<` `<=` `>` `>=` | left |
| 2 | `&&` | left |
| 1 | `\|\|` | left |

##### Arithmetic operators

| Operator | Name | Types | Result |
|----------|------|-------|--------|
| `+` | addition | number + number | number |
| `+` | concatenation | string + string | string |
| `-` | subtraction | number - number | number |
| `*` | multiplication | number * number | number |
| `/` | division | number / number | number |
| `%` | modulo | number % number | number |

Note: Division by zero results in a runtime error. The `+` operator performs addition for numbers and concatenation for strings. No implicit type conversion occurs - `"hello" + 42` is an error.

##### Comparison operators

| Operator | Name | Types | Result |
|----------|------|-------|--------|
| `==` | equal | any == any | boolean |
| `!=` | not equal | any != any | boolean |
| `<` | less than | any < any | boolean |
| `<=` | less than or equal | any <= any | boolean |
| `>` | greater than | any > any | boolean |
| `>=` | greater than or equal | any >= any | boolean |

Note: Comparison operators can compare values of any type. See [Comparison Semantics](#comparison-semantics) for details on how different types are compared.

##### Logical operators

| Operator | Name | Description |
|----------|------|-------------|
| `&&` | logical AND | returns right operand if left is truthy, else left |
| `\|\|` | logical OR | returns left operand if truthy, else right |

Note: Logical operators use short-circuit evaluation and return the actual operand value, not a boolean.

**Special `||` behavior**: When the left operand results in a property-not-found or index-out-of-bounds error, `||` treats this as a falsy value and evaluates the right operand instead of propagating the error.

Examples:

```js
"foo" && "bar"     // "bar" (returns right when left is truthy)
null && "bar"      // null (returns left when left is falsy)
"foo" || "bar"     // "foo" (returns left when left is truthy)
false || "default" // "default" (returns right when left is falsy)

// Special || error handling
obj.missing || "default"    // "default" (missing property treated as falsy)
array[999] || "fallback"    // "fallback" (out of bounds treated as falsy)
obj.exists || "default"     // obj.exists value
```

##### Template expressions

Template expressions allow embedding expressions within string literals using the `${{ }}` syntax. Only double-quoted strings support templates.

```ebnf
template = "${{" Expression "}}" .
```

The expression inside the template must evaluate to a string at runtime. Non-string values will result in a runtime error. `\${{` can be used to escape template expressions.

Examples:

```js
// Simple variable interpolation
"Hello, ${{ name }}!"                    // "Hello, Alice!"
'Welcome ${{ user }}'                    // "Welcome Bob"

// Expressions with operators
"Full name: ${{ firstName + " " + lastName }}"
"Path: ${{ dir }}/${{ file }}"

// Multiple templates in one string
"${{ greeting }}, ${{ name }}! Today is ${{ day }}."

// Complex expressions
"User: ${{ user.firstName }} (${{ user.role }})"
"Items: ${{ items[0] }}, ${{ items[1] }}"

// Escape template
"Hello, \${{ \"world!\" }}"             // "Hello, ${{ \"world!\" }}"

// Errors - expression must return string
"Count: ${{ 42 }}"                      // Error: number not string
"Total: ${{ price + tax }}"             // Error: number not string
```

##### Truthiness

The following values are considered "falsy":

- `false`
- `null`
- `0` (number zero)
- `""` (empty string)
- `[]` (empty array)
- `{}` (empty object)

All other values are considered "truthy".

##### Operator precedence

The precedence of operators is reflected in the grammar. From lowest to highest:

1. `||` (logical OR)
2. `&&` (logical AND)
3. `==`, `!=`, `<`, `<=`, `>`, `>=` (comparison)
4. `+`, `-` (addition, subtraction)
5. `*`, `/`, `%` (multiplication, division, modulo)
6. `+`, `-`, `!` (unary operators)
7. `.`, `[]`, `()` (postfix operators)
