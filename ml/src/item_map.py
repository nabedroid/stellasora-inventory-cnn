# アイテムの ID ⇔ 名前 対応表 (config/items.json) の読み込み。
# 対応表の定義そのものは利用者が管理する。ここは読み込みと変換の提供だけ。
# 形式: [{"id": 1, "name": "青空旅券"}, ...]
# 名前か ID に重複があったら、その場でエラーにして止める (静かにズレるのが一番悪い)。

import json


def load_item_map(path):
  # 対応表を読んで (名前→ID, ID→名前) の辞書ペアを返す
  with open(path, encoding="utf-8") as f:
    entries = json.load(f)
  name_to_id = {}
  id_to_name = {}
  for entry in entries:
    item_id = int(entry["id"])
    name = entry["name"]
    if name in name_to_id:
      raise ValueError(f"対応表で名前が重複しています: {name}")
    if item_id in id_to_name:
      raise ValueError(f"対応表で ID が重複しています: {item_id}")
    name_to_id[name] = item_id
    id_to_name[item_id] = name
  return name_to_id, id_to_name
