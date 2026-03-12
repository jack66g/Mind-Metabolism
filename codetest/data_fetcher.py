import requests
import time
import os

def fetch_learning_data(save_path="training_corpus.txt", target_count=100):
    """
    独立的语料抓取器。
    从网络获取数据并追加存入本地文件，供认知底座离线学习。
    """
    print(f"🚀 开始抓取世界语料，目标新增 {target_count} 条...")
    
    # 建立记忆集合，避免抓取到重复的句子
    collected = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                collected.add(line.strip())
                
    start_count = len(collected)
    added = 0

    # a 模式追加写入
    with open(save_path, 'a', encoding='utf-8') as f:
        while added < target_count:
            try:
                # 这里依然使用你原来的一言 API 作为数据源，后续你可以随意换成别的
                res = requests.get("https://v1.hitokoto.cn/", timeout=3)
                
                if res.status_code == 200:
                    text = res.json().get('hitokoto', '').strip()
                    
                    # 基础清洗与去重
                    if text and len(text) > 3 and text not in collected:
                        collected.add(text)
                        f.write(text + '\n')
                        f.flush() # 实时存盘，防止中途断电丢失
                        added += 1
                        print(f"📥 [入库 {added}/{target_count}] {text}")
                
                # 礼貌抓取间隔，防止被 API 封禁 IP
                time.sleep(1.5)
                
            except Exception as e:
                print(f"⚠️ 网络波动，稍后重试: {e}")
                time.sleep(3)

    print(f"\n✅ 抓取完成！本次成功补充 {added} 条新鲜语料。")
    print(f"📁 语料库总储量: {len(collected)} 条。文件路径: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # 你可以随时运行这个脚本来囤积弹药
    # 比如先抓 200 条存着
    fetch_learning_data(target_count=200000)