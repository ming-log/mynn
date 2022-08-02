# 类型系统

> Python运行时并不强制标注函数和变量类型。类型标注可被用于第三方工具，比如类型检查器、集成开发环境、静态检查器等。

最基本的支持由`Any`，`Union`，`Tuple`，`Callable`，`TypeVar`和`Generic`类型组成。

函数接受并返回一个字符串，注释像下面这样：

```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```

在函数 `greeting` 中，参数 `name` 预期是 `str`类型，并且返回 `str`类型。子类型允许作为参数。