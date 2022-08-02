# -*- coding: utf-8 -*-
"""
@Time:2022/8/2 10:03
@Author:Ming-Log
@File:Python类型系统.py
@IDE:PyCharm
"""
# 函数接受并返回一个字符串
def greeting(name: str) -> str:
    return 'hello' + name

greeting(1)  # 发现出现了，编译检查错误，应为类型str，但实际为int
greeting('213') + 1  # 发现出现了，编译检查错误，应为类型str，但实际为int


# 类型别名
# 类型别名通过将类型分配给别名来定义。在这个例子中，Vector和 List[float] 将被视为可互换的同义词
from typing import List
Vector = List[float]  # 元素类型为float的列表

def scale(scale: float, vector: Vector) -> Vector:
    return [scale * num for num in vector]

new_vector = scale(2.0, [2, 2, 2])

# 类型别名可用于简化复杂类型签名。
from typing import Dict, Tuple, Sequence

ConnectionOptions = Dict[str, str]  # 字典类型，键值对均为字符串
Address = Tuple[str, int]
Server = Tuple[Address, ConnectionOptions]


# 无返回值的时候可指定返回类型为None
def broadcast_message(
        message: str,
        servers: Sequence[Tuple[Tuple[str, int], Dict[str, str]]]) -> None:
    ...

msg = '123'
sev = Sequence[(('hello', 123), {'123': '123'})]
broadcast_message(msg, sev)

# NewType
# 使用 NewType() 辅助函数创建不同的类型:
from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(524313)
# 静态类型检查器会将新类型视为它是原始类型的子类。这对于帮助捕捉逻辑错误非常有用:
def get_user_name(user_id: UserId) -> str:
    ...

# typechecks
user_a = get_user_name(UserId(42351))
user_b = get_user_name(-1)  # 没有通过检查，类型应该为UserID，不应该为int

# 仍然可以对 UserId 类型的变量执行所有的 int 支持的操作，但结果将始终为 int 类型。
# 这可以让你在需要 int 的地方传入 UserId，但会阻止你以无效的方式无意中创建 UserId:
# 'output' is of type 'int', not 'UserId'
output = UserId(23413) + UserId(54341)


from typing import NewType

UserId = NewType('UserId', int)

# 运行失败，但是类型检查无问题
class AdminUserId(UserId): ...


