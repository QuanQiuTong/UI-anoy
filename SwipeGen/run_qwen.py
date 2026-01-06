from app_explorer import AppExplorer

def main():
    PACKAGES2 = {
        # "SHEIN":"com.zzkko",
        # "deepseek":"com.deepseek.chat",
        # "网易云音乐": "com.netease.cloudmusic",
        # "京东": "com.jingdong.app.mall",
        "QQ": "com.tencent.mobileqq",
        "哔哩哔哩": "tv.danmaku.bili",
        # "知乎": "com.zhihu.android",
        # "美团": "com.sankuai.meituan"
    }

    PACKAGES3 = {
        "Youtube": "com.google.android.youtube",
        "Google Maps": "com.google.android.apps.maps",
        "Discord": "com.discord",
        "Zoom": "us.zoom.videomeetings",
        "Whatsapp": "com.whatsapp",
    }

    PACKAGE_NEW = {
        # "viggle.ai": "com.warpengine.viggle",
        # "gemini": "com.google.android.googlequicksearchbox",
        # "wiser": "com.wisernow.android",
        # "finch": "com.finch.finch",
        # "manus": "tech.butterfly.app",

        # "Perplexity Comet": "ai.perplexity.comet",
        # "Perplexity AI": "ai.perplexity.app.android",
        # "Arc Search": "company.thebrowser.arc",
        # "Focus Friend": "com.underthing.focus.friend",
        # "Pingo AI": "com.picoai.languageapp.android",

        "Perch Reader": "com.gmlabs.perch",
        "OmniTools": "com.snitl.omnitools",
        "Stellarium": "com.noctuasoftware.stellarium_free",
        "Jagat": "io.jagat.lite",
        "Bluesky": "xyz.blueskyweb.app",
        "Arts & Culture": "com.google.android.apps.cultural",
    }

    PACKAGE_HOT = {
        "TikTok": "com.zhiliaoapp.musically",
        "Instagram": "com.instagram.android",
        "Facebook": "com.facebook.katana",
        "Twitter": "com.twitter.android",
        "Snapchat": "com.snapchat.android",
    }
    
    # 确保 backend server 已启动
    model_url = "http://127.0.0.1:8000/" 
    
    for app in PACKAGE_HOT.values():
        explorer = AppExplorer(None, model_url, app)
        explorer.explore_app(max_l1_clicks=5, max_l2_interactions=3)

if __name__ == "__main__":
    main()