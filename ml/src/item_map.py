# アイテムの ID ⇔ 名前 対応をマスター辞書 (master-dictionary.json) から引く。
# 旧 config/items.json の対応表は廃止し、マスター辞書に一本化した。
# ここでは各エントリの id / name だけを使う (iconDataUrl や features は見ない)。
# 名前か ID に重複があったら、その場でエラーにして止める (静かにズレるのが一番悪い)。

import json


def load_item_map(master_path):
  # マスター辞書を読んで (名前→ID, ID→名前) の辞書ペアを返す
  with open(master_path, encoding="utf-8") as f:
    entries = json.load(f)
  name_to_id = {}
  id_to_name = {}
  for entry in entries:
    item_id = int(entry["id"])
    name = entry["name"]
    if name in name_to_id:
      raise ValueError(f"マスター辞書で名前が重複しています: {name}")
    if item_id in id_to_name:
      raise ValueError(f"マスター辞書で ID が重複しています: {item_id}")
    name_to_id[name] = item_id
    id_to_name[item_id] = name
  return name_to_id, id_to_name
