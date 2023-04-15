
class Tokeniser:
  def __init__(self):
    self.AA = {
    "A" : 0,
    "R" : 1,
    "N" : 2,
    "D" : 3,
    "C" : 4,
    "Q" : 5,
    "E" : 6,
    "G" : 7,
    "H" : 8,
    "I" : 9,
    "L" : 10,
    "K" : 11,
    "M" : 12,
    "F" : 13,
    "P" : 14,
    "S" : 15,
    "T" : 16,
    "W" : 17,
    "Y" : 18,
    "V" : 19,
    "*" : 20,
    "-" : 21,
    "?" : 22,

    }
    self.inverse_AA = {
      0 : "A",
      1 : "R",
      2 : "N",
      3 : "D",
      4 : "C",
      5 : "Q",
      6 : "E",
      7 : "G",
      8 : "H",
      9 : "I",
      10 : "L",
      11 : "K",
      12 : "M",
      13 : "F",
      14 : "P",
      15 : "S",
      16 : "T",
      17 : "W",
      18 : "Y",
      19 : "V",
      20 : "*",
      21 : "-",
      22 : "?",
    }

  def token_to_string(self, tokens):
    aa = ''
    for t in tokens:
      aa += self.inverse_AA[t.item()]
    return aa

  def string_to_token(self, string):
    aa = []
    for char in string:
      aa.append(self.AA[char])
    return aa

