"""
LESSON 1 — Python for someone who already knows JavaScript/Node
Run me:  .venv/bin/python lessons/l1_python_for_node_devs.py

Read the comments top to bottom. Everything here is stuff you'll actually
use in the project. Nothing extra.
"""

# ---------------------------------------------------------------
# 1. NO BRACES. Indentation IS the block. 4 spaces. That's the rule.
# ---------------------------------------------------------------
# JS:  function greet(name) { return `hi ${name}` }
def greet(name):
    return f"hi {name}"          # f"..." is a template string, like `...` in JS

print(greet("Amrendra"))


# ---------------------------------------------------------------
# 2. Variables / types  (no let, no const, no var — just name = value)
# ---------------------------------------------------------------
count = 5                        # int
price = 9.99                     # float
name = "docs"                    # str
is_ready = True                  # note: True/False capitalised, not true/false
nothing = None                   # this is JS's null/undefined

# type hints are OPTIONAL and only for humans + editors. Python ignores them.
# But FastAPI USES them to validate requests, so you'll write them a lot.
def add(a: int, b: int) -> int:
    return a + b

print(add(2, 3))

def add_two_number(a : int, b : int) -> int:
    return a + b;

print(add_two_number(2, 3));

# ---------------------------------------------------------------
# 3. Lists (JS array) and dicts (JS object)
# ---------------------------------------------------------------
items = ["pdf", "png", "txt"]        # list  -> JS array
items.append("md")                   # .push()
print(items[0], items[-1], len(items))   # [-1] = last item. len() not .length

user = {"name": "avs", "role": "dev"}    # dict -> JS object
print(user["name"])                       # NO dot access. Always ["key"]
print(user.get("email", "not set"))       # .get() = safe, gives default instead of crashing

for key, value in user.items():
    print(" ", key, "=", value)


# ---------------------------------------------------------------
# 4. List comprehension — this is the one thing that looks alien.
#    It's just .map() / .filter() written backwards.
# ---------------------------------------------------------------
nums = [1, 2, 3, 4, 5]

doubled = [n * 2 for n in nums]                 # JS: nums.map(n => n*2)
evens   = [n for n in nums if n % 2 == 0]       # JS: nums.filter(n => n%2===0)
print(doubled, evens)

# you'll see this constantly in AI code, e.g.
texts = [chunk.strip() for chunk in "a , b , c".split(",")]
print(texts)


# ---------------------------------------------------------------
# 5. Loops
# ---------------------------------------------------------------
for i in range(3):               # 0,1,2
    print("i =", i)

for idx, item in enumerate(items):     # JS: items.forEach((item, idx) => ...)
    print(idx, item)


# ---------------------------------------------------------------
# 6. Errors — try/except (not try/catch)
# ---------------------------------------------------------------
try:
    broken = 1 / 0
except ZeroDivisionError as e:
    print("caught it:", e)
finally:
    print("this always runs")


# ---------------------------------------------------------------
# 7. Classes — you mostly won't write these, but FastAPI models look like them
# ---------------------------------------------------------------
class Chunk:
    def __init__(self, text, source):   # __init__ = constructor
        self.text = text                # 'self' = 'this', and you MUST type it
        self.source = source

    def preview(self):
        return self.text[:20]           # slicing: first 20 chars

c = Chunk("hello this is a long piece of text", "notes.pdf")
print(c.preview(), "|", c.source)


# ---------------------------------------------------------------
# 8. Imports + "main guard"
# ---------------------------------------------------------------
import os
import json

print("cwd:", os.getcwd())
print("json:", json.dumps({"ok": True}))

# This line means: "only run this when the file is executed directly,
# not when someone imports it." Same idea as require.main === module.
if __name__ == "__main__":
    print("\n✅ Lesson 1 done. You now know 90% of the Python this project needs.")


# ---------------------------------------------------------------
# CHEAT MAP — JS  ->  Python
# ---------------------------------------------------------------
# const/let          -> just x = 1
# `hi ${x}`          -> f"hi {x}"
# array              -> list  []
# object             -> dict  {}
# obj.key            -> obj["key"]
# .length            -> len(x)
# .push()            -> .append()
# .map()             -> [f(x) for x in list]
# .filter()          -> [x for x in list if cond]
# null / undefined   -> None
# true / false       -> True / False
# ===                -> ==   (python's == already compares by value)
# &&  ||  !          -> and  or  not
# try/catch          -> try/except
# npm install        -> pip install
# package.json       -> requirements.txt
# node_modules/      -> .venv/
# module.exports     -> (nothing — every top-level name is exported)
# require('x')       -> import x
