
class Tokeniser:
  def __init__(self):
    self.AA = {
    "-" : 0,
    "A" : 1,
    "R" : 2,
    "N" : 3,
    "D" : 4,
    "C" : 5,
    "Q" : 6,
    "E" : 7,
    "G" : 8,
    "H" : 9,
    "I" : 10,
    "L" : 11,
    "K" : 12,
    "M" : 13,
    "F" : 14,
    "P" : 15,
    "S" : 16,
    "T" : 17,
    "W" : 18,
    "Y" : 19,
    "V" : 20,
    "*" : 21,
    "?" : 22,

    }
    self.inverse_AA = {
      0 : "-",
      1 : "A",
      2 : "R",
      3 : "N",
      4 : "D",
      5 : "C",
      6 : "Q",
      7 : "E",
      8 : "G",
      9 : "H",
      10 : "I",
      11 : "L",
      12 : "K",
      13 : "M",
      14 : "F",
      15 : "P",
      16 : "S",
      17 : "T",
      18 : "W",
      19 : "Y",
      20 : "V",
      21 : "*",
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

