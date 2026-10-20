"""Names of the MediaWiki namespaces that the default crawl exclude patterns skip.

Each wiki shows a namespace under its own language, for example `Spezial:` for special pages on
German Wikipedia. The names come from the siteinfo of the 24 largest Wikipedias that ship in
English, German, French, Spanish, Italian, Portuguese, Russian, Japanese, Chinese, Dutch, Polish,
Swedish, Arabic, Ukrainian, Vietnamese, Turkish, Indonesian, Korean, Czech, Persian, Hungarian,
Finnish, Norwegian and Catalan, with their aliases. A short alias is left out: an ASCII one of
three letters or fewer, or any of one character, because it can start an article title. `Mall`
(Swedish templates) and `Vita` (Hungarian talk pages) are left out too: English Wikipedia has
articles whose titles start with them.
"""

from __future__ import annotations

# fmt: off
PROJECT: tuple[str, ...] = (
    "Project", "Vikipedi", "Wikipedia", "Wikipedie", "Viquipèdia", "Wikipédia", "Википедия",
    "ВП", "Вікіпедія", "شن", "وب", "ويكي_پديا", "ويكيبيديا", "وپ", "ویکی_پدیا", "专案", "專案",
    "維基百科", "维基百科", "위키백과",
)

PORTAL: tuple[str, ...] = (
    "Portaal", "Portail", "Portal", "Portale", "Teemasivu", "Chủ_đề", "Cổng_thông_tin",
    "Portál", "Портал", "بوابة", "درگاه", "主題", "主题", "들머리", "포털",
)

SPECIAL: tuple[str, ...] = (
    "Especial", "Istimewa", "Speciaal", "Special", "Speciale", "Specjalna", "Spesial",
    "Spezial", "Toiminnot", "Speciális", "Speciální", "Spécial", "Özel", "Đặc_biệt",
    "Служебная", "Спеціальна", "Спеціальні", "خاص", "ویژه", "特別", "特殊", "특수", "특수기능",
)

CATEGORY: tuple[str, ...] = (
    "Categoria", "Categorie", "Category", "Kategori", "Kategoria", "Kategorie", "Luokka",
    "Categoría", "Catégorie", "Kategória", "Thể_loại", "Категория", "Категорія", "تصنيف", "رده",
    "カテゴリ", "分类", "分類", "분류",
)

TEMPLATE: tuple[str, ...] = (
    "Malline", "Plantilla", "Sablon", "Sjabloon", "Szablon", "Templat", "Template",
    "Vorlage", "Bản_mẫu", "Modèle", "Predefinição", "Tiêu_bản", "Şablon", "Šab", "Šablona",
    "Шаблон", "الگو", "قالب", "テンプレート", "样板", "模板", "樣板",
)

TEMPLATE_TALK: tuple[str, ...] = (
    "Discussioni_template", "Dyskusja_szablonu", "Keskustelu_mallineesta", "Maldiskusjon",
    "Malldiskussion", "Overleg_sjabloon", "Pembicaraan_Templat", "Sablon_vita", "Sablonvita",
    "Templat_Pembicaraan", "Template_talk", "Vorlage_Diskussion", "Discussion_modèle",
    "Discussion_Modèle", "Discutida_Modèl", "Diskuse_k_šabloně", "Plantilla_discusión",
    "Plantilla_Discussió", "Predefinição_Discussão", "Template‐ノート", "Thảo_luận_Bản_mẫu",
    "Thảo_luận_Tiêu_bản", "Şablon_tartışma", "Šablona_diskuse", "Обговорення_шаблона",
    "Обговорення_шаблону", "Обсуждение_шаблона", "بحث_الگو", "نقاش_القالب",
    "テンプレート・トーク", "样板对话", "样板讨论", "模板对话", "模板對話", "模板討論",
    "模板讨论", "樣板對話", "樣板討論", "틀토론",
)

TALK: tuple[str, ...] = (
    "Bicara", "Discussion", "Discussione", "Discuter", "Diskuse", "Diskusjon", "Diskussion",
    "Dyskusja", "Keskustelu", "Overleg", "Pembicaraan", "Talk", "Discusión",
    "Discussió", "Discussão", "Komentár", "Tartışma", "Thảo_luận", "Обговорення", "Обсуждение",
    "بحث", "نقاش", "トーク", "ノート", "对话", "對話", "討論", "讨论", "토론",
)

FILE: tuple[str, ...] = (
    "Afbeelding", "Archivo", "Arquivo", "Berkas", "Bestand", "Bild", "Bilde", "Datei", "Dosya",
    "Ficheiro", "Fichier", "File", "Fitxer", "Gambar", "Grafika", "Image", "Imagem", "Imagen",
    "Imatge", "Immagine", "Kuva", "Plik", "Resim", "Soubor", "Tiedosto", "Fájl", "Hình", "Kép",
    "Obrázok", "Tập_tin", "Зображення", "Изображение", "Файл", "تصویر", "صورة", "ملف", "پرونده",
    "ファイル", "图像", "图片", "圖像", "圖片", "文件", "档案", "檔案", "画像", "그림", "파일",
)

FILE_TALK: tuple[str, ...] = (
    "Bild_Diskussion", "Bilddiskussion", "Bildediskusjon", "Datei_Diskussion",
    "Discussion_fichier", "Discussion_Fichier", "Discussion_Image", "Discussion_Imatge",
    "Discussioni_file", "Discussioni_immagine", "Discutida_Imatge", "Diskuse_k_souboru",
    "Dyskusja_grafiki", "Dyskusja_pliku", "Fildiskusjon", "Fildiskussion", "File_talk",
    "Gambar_Pembicaraan", "Image_talk", "Keskustelu_kuvasta", "Keskustelu_tiedostosta",
    "Overleg_afbeelding", "Overleg_bestand", "Pembicaraan_Berkas", "Pembicaraan_Gambar",
    "Soubor_diskuse", "Archivo_discusión", "Arquivo_Discussão", "Diskusia_k_obrázku",
    "Dosya_tartışma", "Ficheiro_Discussão", "Fitxer_Discussió", "Fájlvita", "Imagem_Discussão",
    "Imagen_discusión", "Imatge_Discussió", "Komentár_k_obrázku", "Kép_vita", "Képvita",
    "Resim_tartışma", "Thảo_luận_Hình", "Thảo_luận_Tập_tin", "Обговорення_зображення",
    "Обговорення_файлу", "Обсуждение_изображения", "Обсуждение_файла", "بحث_تصویر",
    "بحث_پرونده", "نقاش_الصورة", "نقاش_الملف", "ファイル‐ノート", "ファイル・トーク",
    "图像对话", "图像讨论", "图片讨论", "圖像對話", "圖像討論", "圖片討論", "文件对话",
    "文件對話", "文件討論", "文件讨论", "档案对话", "档案讨论", "檔案對話", "檔案討論",
    "画像‐ノート", "그림토론", "파일토론",
)

USER: tuple[str, ...] = (
    "Benutzer", "Benutzerin", "Bruker", "Gebruiker", "Pengguna", "Redaktor", "User", "Usuari",
    "Usuaria", "Usuario", "Utente", "Utilisateur", "Utilisator", "Utilisatrice", "Utilizador",
    "Utilizador(a)", "Utilizadora", "Wikipedista", "Wikipedistka", "Wikipedysta",
    "Wikipedystka", "Användare", "Kullanıcı", "Käyttäjä", "Szerkesztő", "Thành_viên", "Usuària",
    "Usuária", "Usuário", "Usuário(a)", "Uživatel", "Užívateľ", "Užívateľka", "Користувач",
    "Користувачка", "Участник", "Участница", "كاربر", "مستخدم", "مستخدمة", "کاربر", "使用者",
    "利用者", "用戶", "用户", "사용자",
)

USER_TALK: tuple[str, ...] = (
    "Benutzer_Diskussion", "Benutzerin_Diskussion", "Bicara_Pengguna", "Brukerdiskusjon",
    "Discussion_Utilisateur", "Discussion_utilisateur", "Discussion_Utilisator",
    "Discussion_utilisatrice", "Discussion_Utilisatrice", "Discussioni_utente",
    "Discutida_Utilisator", "Diskuse_s_wikipedistkou", "Diskuse_s_wikipedistou",
    "Diskusia_s_redaktorom", "Dyskusja_wikipedystki", "Dyskusja_wikipedysty",
    "Overleg_gebruiker", "Pembicaraan_Pengguna", "User_talk", "User_vita",
    "Wikipedista_diskuse", "Wikipedistka_diskuse", "Användardiskussion", "Diskuse_s_uživatelem",
    "Diskusia_s_užívateľkou", "Diskusia_s_užívateľom", "Keskustelu_käyttäjästä",
    "Komentár_k_redaktorovi", "Kullanıcı_mesaj", "Szerkesztővita", "Thảo_luận_Thành_viên",
    "Usuari_Discussió", "Usuaria_discusión", "Usuario_discusión", "Usuària_Discussió",
    "Usuária_Discussão", "Usuário(a)_Discussão", "Usuário_Discussão", "Utilizador(a)_Discussão",
    "Utilizador_Discussão", "Utilizadora_Discussão", "Uživatel_diskuse", "Uživatelka_diskuse",
    "Обговорення_користувача", "Обговорення_користувачки", "Обсуждение_участника",
    "Обсуждение_участницы", "ОУ", "بحث_كاربر", "بحث_کاربر", "بک", "نقاش_المستخدم",
    "نقاش_المستخدمة", "使用者对话", "使用者對話", "使用者討論", "使用者讨论", "利用者‐会話",
    "利用者・トーク", "用戶對話", "用戶討論", "用户对话", "用户讨论", "사용자토론", "사토",
)
# fmt: on
