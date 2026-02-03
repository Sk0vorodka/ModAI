import asyncio
import logging
import io
import re
import html
import math
import difflib
import os
from typing import Optional, Dict, List, Any, Tuple

# Убедитесь, что файл prompts.py лежит в той же папке
from prompts import PROMPT_HIKKA_GEN, PROMPT_HIKKA_FIX, PROMPT_EXTERA_GEN, PROMPT_EXTERA_FIX

import aiohttp
import aiosqlite
from dotenv import load_dotenv
from aiohttp_socks import ProxyConnector 

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
    FSInputFile,
    Message
)

# --- CONFIG ---
load_dotenv() 

BOT_TOKEN = os.getenv("BOT_TOKEN")
ONLYSQ_KEY_DEFAULT = os.getenv("ONLYSQ_KEY", "openai")
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))
PROXY_URL = os.getenv("PROXY_URL") 

DB_NAME = "bot_database.db"
MAX_FILE_SIZE = 1024 * 500
MAX_TOKENS = 20000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

http_session_direct: Optional[aiohttp.ClientSession] = None
http_session_proxy: Optional[aiohttp.ClientSession] = None

# --- DIFF SYSTEM CONSTANTS ---
PROMPT_DIFF_ADDON = (
    "\n\n⚡️ FAST EDIT MODE:\n"
    "You generally should NOT rewrite the entire file. Only output the changes.\n"
    "Use this format exactly to replace code blocks:\n"
    "<<<<<<< SEARCH\n"
    "    original line 1\n"
    "    original line 2\n"
    "=======\n"
    "    new line 1\n"
    "    new line 2\n"
    ">>>>>>>\n"
    "\n"
    "Rules:\n"
    "1. The SEARCH block must match the original code EXACTLY (indentation, spaces).\n"
    "2. If you need to replace multiple parts, use multiple SEARCH/REPLACE blocks.\n"
    "3. If the file is small or changes are massive, you MAY output the full file inside ```python ... ```."
    "if you generating a plugin, DONT USE THE CACTUSLIB!!"
)

# --- MODELS ---
PROVIDERS_CONFIG = {
    "onlysq": {
        "name": "OnlySq", "icon": "🔶", "base_url": "https://api.onlysq.ru/ai/openai",
        "models": {
            "grok-3": {"name": "Grok 3", "icon": "🚀", "desc": "Мощная модель от xAI."},
            "gpt-5": {"name": "GPT-5", "icon": "🤯", "desc": "Next-Gen OpenAI."},
            "qwen-3-32b": {"name": "Qwen 3", "icon": "💪", "desc": "Мощная модель для кодинга"},
            "gpt-4o": {"name": "GPT-4o", "icon": "🧠", "desc": "Стабильная классика."},
            "deepseek-r1": {"name": "Deepseek r1", "icon": "⚡", "desc": "Рассуждающая модель."},
            "gpt-5.2-chat": {"name": "GPT-5 Chat", "icon": "🤯", "desc": "Latest from OpenAI."},
            "o3": {"name": "o3", "icon": "🧠", "desc": "Очень умная"},
            "o4-mini": {"name": "o4 mini", "icon": "🧠", "desc": "Немного умнее чем o3"},
        }
    },
    "gemini": {
        "name": "Gemini", "icon": "💎", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": {
            "gemini-pro-latest": {"name": "Gemini Pro Latest", "icon": "🌌", "desc": "Последняя модель Pro версии."},
            "gemini-flash-latest": {"name": "Gemini Flash Latest", "icon": "🌌", "desc": "Последняя модель Flash версии."},
            "gemini-3-pro-preview": {"name": "Gemini 3 Pro", "icon": "🌌", "desc": "Третье поколение Pro версии."},
            "gemini-3-flash-preview": {"name": "Gemini 3 Flash", "icon": "🌌", "desc": "Третье поколение Flash версии."},
            "gemini-2.5-pro": {"name": "Gemini 2.5 Pro", "icon": "💎", "desc": "Мощная и точная."},
            "gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "icon": "⚡", "desc": "Скоростная."},
            "gemini-2.5-flash-lite": {"name": "Gemini 2.5 Lite", "icon": "🔦", "desc": "Flashlight версия."},
        }
    },
    "openai": {
        "name": "OpenAI", "icon": "🤖", "base_url": "https://api.openai.com/v1",
        "models": {
            "gpt-5": {"name": "GPT-5", "icon": "🤯", "desc": "Новейшая модель."},
            "gpt-5-turbo": {"name": "GPT-5 Turbo", "icon": "🚀", "desc": "Ускоренная GPT-5."},
            "gpt-4o": {"name": "GPT-4o", "icon": "🧠", "desc": "Омни-модель."},
            "gpt-4o-mini": {"name": "GPT-4o Mini", "icon": "⚡", "desc": "Мини."}
        }
    },
    "openrouter": {
        "name": "OpenRouter", "icon": "🌐", "base_url": "https://openrouter.ai/api/v1",
        "models": {
            "tngtech/deepseek-r1t2-chimera:free": {"name": "DeepSeek R1T2 Chimera", "icon": "🆓", "desc": "Бесплатная R1T2."},
            "nvidia/nemotron-3-nano-30b-a3b:free": {"name": "Nemotron 3 Nano", "icon": "🆓", "desc": "Открытая модель от NVIDIA."},
            "google/gemma-3-27b-it:free": {"name": "Gemma 3 27B", "icon": "🆓", "desc": "Gemma Free."},
            "upstage/solar-pro-3:free": {"name": "Solar Pro 3", "icon": "🆓", "desc": "Мощная модель Upstage."},
        }
    }
}

# --- DB ---
async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, username TEXT, model TEXT DEFAULT 'gpt-4o-mini')")
        cols = ["gemini_key", "openai_key", "openrouter_key", "onlysq_key", "provider"]
        for c in cols:
            try: await db.execute(f"ALTER TABLE users ADD COLUMN {c} TEXT")
            except: pass
        
        await db.execute("CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, username TEXT, p_type TEXT, prompt TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pending_gens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                chat_id INTEGER,
                sys_prompt TEXT,
                user_prompt TEXT,
                ext TEXT,
                is_fix INTEGER,
                original_code TEXT
            )
        """)
        
        await db.commit()

async def get_user_settings(user_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute("SELECT gemini_key, openai_key, openrouter_key, onlysq_key, model, provider FROM users WHERE user_id = ?", (user_id,)) as c:
            r = await c.fetchone()
            if r: return {"gemini_key": r[0], "openai_key": r[1], "openrouter_key": r[2], "onlysq_key": r[3], "model": r[4], "provider": r[5] or "onlysq"}
            return {"gemini_key": None, "openai_key": None, "openrouter_key": None, "onlysq_key": None, "model": "gpt-4o-mini", "provider": "onlysq"}

async def update_user(user_id, username, **kwargs):
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,)) as c:
            if not await c.fetchone(): await db.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, username))
        for k, v in kwargs.items():
            val = None if v == "RESET" else v
            await db.execute(f"UPDATE users SET {k} = ? WHERE user_id = ?", (val, user_id))
        await db.commit()

async def add_pending_gen(user_id, chat_id, sys, prompt, ext, is_fix, original_code):
    async with aiosqlite.connect(DB_NAME) as db:
        cursor = await db.execute(
            "INSERT INTO pending_gens (user_id, chat_id, sys_prompt, user_prompt, ext, is_fix, original_code) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, chat_id, sys, prompt, ext, 1 if is_fix else 0, original_code)
        )
        await db.commit()
        return cursor.lastrowid

async def remove_pending_gen(row_id):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("DELETE FROM pending_gens WHERE id = ?", (row_id,))
        await db.commit()

async def get_all_pending_gens():
    async with aiosqlite.connect(DB_NAME) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM pending_gens") as cursor:
            return await cursor.fetchall()

# --- UTILS (KEYBOARDS) ---

def get_main_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 Ген. Модуль", callback_data="nav_gen_mod"), InlineKeyboardButton(text="🛠 Фикс Модуля", callback_data="nav_fix_mod")],
        [InlineKeyboardButton(text="🧩 Ген. Плагин", callback_data="nav_gen_plug"), InlineKeyboardButton(text="🔧 Фикс Плагина", callback_data="nav_fix_plug")],
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="nav_main_settings")]
    ])

def get_cancel_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔙 Отмена", callback_data="cancel")]
    ])

# --- LOGIC ---
class GenStates(StatesGroup):
    generating = State()
    waiting_for_key = State()
    waiting_for_gen_mod = State()
    waiting_for_fix_mod_file = State()
    waiting_for_fix_mod_prompt = State()
    waiting_for_gen_plug = State()
    waiting_for_fix_plug_file = State()
    waiting_for_fix_plug_prompt = State()

async def _api_request(sys, user, user_id):
    s = await get_user_settings(user_id)
    prov = s["provider"]
    conf = PROVIDERS_CONFIG.get(prov, PROVIDERS_CONFIG["onlysq"])
    
    # 1. Получаем строку с ключами (приоритет: БД пользователя -> ENV переменная)
    user_keys = s.get(f"{prov}_key")
    if user_keys and len(user_keys.strip()) > 5:
         # Если в базе есть ключи - используем их
        key_data = user_keys
    else:
        # Иначе пробуем дефолтный ключ (только для OnlySQ)
        key_data = os.getenv("ONLYSQ_KEY") if prov == "onlysq" else None

    if not key_data: 
        return f"ERROR: Ключи для провайдера '{prov}' не установлены в настройках, а дефолтный ключ отсутствует."
    
    # 2. Разбиваем ключи (учитываем любые разделители для надежности)
    api_keys = [k.strip() for k in key_data.split('\n') if k.strip()]
    
    url = conf["base_url"]
    if not url.endswith("/chat/completions"):
        if url.endswith("/"): url = url[:-1]
        url = f"{url}/chat/completions"
        
    data = {"model": s["model"], "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}], "max_tokens": MAX_TOKENS}
    
    # Выбор сессии
    if prov == "onlysq":
        current_session = http_session_direct
    else:
        current_session = http_session_proxy if http_session_proxy else http_session_direct

    last_error = ""
    success = False
    
    # --- ЦИКЛ ПЕРЕБОРА КЛЮЧЕЙ ---
    for index, current_key in enumerate(api_keys):
        headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
        if prov == "openrouter":
            headers["HTTP-Referer"] = "https://t.me/AiModuleBot"
            headers["X-Title"] = "ModAI Bot"

        try:
            async with current_session.post(url, headers=headers, json=data, timeout=300) as resp:
                if resp.status == 200: 
                    res = await resp.json()
                    return res["choices"][0]["message"]["content"]
                
                err = await resp.text()
                
                # Логируем ошибку, но пробуем следующий ключ
                last_error = f"Key #{index+1} ({prov}) Err: {resp.status} - {err[:100]}"
                logger.warning(last_error)

                # Если ошибка фатальная (400 - Bad Request, 404 - Not Found), нет смысла перебирать ключи
                if resp.status in [400, 404]:
                    return f"ERROR: Критическая ошибка API ({resp.status}): {err[:200]}"
                
                # Для 429 (лимиты), 401 (плохой ключ), 403 (бан/доступ) -> идем дальше
                continue 
                
        except Exception as e:
            last_error = f"Connection Err: {e}"
            continue 

    return f"ERROR: Все ключи ({len(api_keys)} шт.) для провайдера '{prov}' не сработали. Последняя ошибка: {last_error}"

async def safe_delete(bot: Bot, chat_id: int, message_id: int):
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass 

# --- NEW: DIFF APPLY LOGIC ---
def apply_patch(original_code: str, response_text: str) -> Tuple[str, str]:
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()
    patch_pattern = r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>"
    
    def apply_diffs(target_code, source_text):
        matches = list(re.finditer(patch_pattern, source_text, re.DOTALL))
        if not matches:
            return target_code, 0, []
        new_code = target_code
        applied_count = 0
        errors = []
        for match in matches:
            search_block = match.group(1)
            replace_block = match.group(2)
            if search_block in new_code:
                new_code = new_code.replace(search_block, replace_block, 1)
                applied_count += 1
            elif search_block.strip() and search_block.strip() in new_code:
                new_code = new_code.replace(search_block.strip(), replace_block, 1)
                applied_count += 1
            else:
                errors.append(f"Не найден фрагмент: {search_block[:30]}...")
        return new_code, applied_count, errors

    code_block_pattern = r"```(?:python|py|plugin)?\s*(.*?)```"
    code_blocks = list(re.finditer(code_block_pattern, text, re.DOTALL))
    
    extracted_content = None
    if code_blocks:
        extracted_content = code_blocks[-1].group(1)

    if extracted_content:
        if re.search(r"<<<<<<< SEARCH", extracted_content):
            new_code, count, errs = apply_diffs(original_code, extracted_content)
            status = f"Применено {count} правок (из блока кода)."
            if errs: status += f" Не найдено: {len(errs)}."
            comment = re.sub(code_block_pattern, "", text, flags=re.DOTALL).strip()
            return new_code, f"{status}\n\n{comment}"
        else:
            comment = re.sub(code_block_pattern, "", text, flags=re.DOTALL).strip()
            return extracted_content.strip(), (comment if comment else "Полная перезапись (найден блок кода).")

    if re.search(r"<<<<<<< SEARCH", text):
        new_code, count, errs = apply_diffs(original_code, text)
        if count > 0:
            comment = re.sub(patch_pattern, "", text, flags=re.DOTALL).strip()
            status = f"Применено {count} правок (Raw Text)."
            if errs: status += f" Не найдено: {len(errs)}."
            return new_code, f"{status}\n\n{comment}"

    return text.strip(), "Код получен (без блоков и патчей)."

# --- HANDLERS (START & MENU) ---

@router.message(GenStates.generating)
async def busy(m: Message): pass

@router.message(CommandStart())
async def start(m: Message, state: FSMContext):
    await update_user(m.from_user.id, m.from_user.username)
    await state.clear()
    await m.answer("<a href='tg://emoji?id=5222108309795908493'>👋</a> <b>AiGen Bot</b>", reply_markup=get_main_kb(), parse_mode='HTML')

@router.callback_query(F.data == "cancel")
async def cancel(c: types.CallbackQuery, state: FSMContext):
    await state.clear()
    try:
        await c.message.edit_text("<a href='tg://emoji?id=5222108309795908493'>👋</a> <b>AiGen Bot</b>", reply_markup=get_main_kb(), parse_mode='HTML')
    except Exception:
        await c.message.answer("<a href='tg://emoji?id=5222108309795908493'>👋</a> <b>AiGen Bot</b>", reply_markup=get_main_kb(), parse_mode='HTML')
    await c.answer()

# --- HANDLERS (SETTINGS) ---

async def show_tab(event, active):
    user_id = event.from_user.id
    s = await get_user_settings(user_id)
    conf = PROVIDERS_CONFIG[active]
    
    # Подсчет сохраненных ключей для отображения статуса
    keys_stored = s.get(f"{active}_key", "")
    key_count = len([k for k in keys_stored.split('\n') if k.strip()]) if keys_stored else 0
    key_status = f"✅ ({key_count})" if key_count > 0 else "❌"

    text = f"<a href='tg://emoji?id=5301096984617166561'>🤖</a> <b>Выбор модели:</b>\n\n{conf['icon']} <b>{conf['name']}</b> (Ключи: {key_status}):\n"
    for _, m in conf["models"].items(): 
        text += f"• {m['name']} — {m['desc']}\n"
    
    btns = []
    row1 = [InlineKeyboardButton(text=f"{PROVIDERS_CONFIG[p]['icon']} {PROVIDERS_CONFIG[p]['name']}", callback_data=f"tab:{p}") for p in ["onlysq", "gemini"]]
    row2 = [InlineKeyboardButton(text=f"{PROVIDERS_CONFIG[p]['icon']} {PROVIDERS_CONFIG[p]['name']}", callback_data=f"tab:{p}") for p in ["openai", "openrouter"]]
    btns.extend([row1, row2, [InlineKeyboardButton(text=f"——— {conf['icon']} {conf['name']} ———", callback_data="ignore")]])
    
    models_keys = list(conf["models"].keys())
    for i, mid in enumerate(models_keys):
        m_data = conf["models"][mid]
        # Проверяем, выбран ли этот провайдер И эта модель
        is_selected = (s["model"] == mid and s["provider"] == active)
        mark = "✅" if is_selected else m_data["icon"]
        btns.append([InlineKeyboardButton(text=f"{mark} {m_data['name']}", callback_data=f"sm:{active}:{i}")])
    
    btns.append([InlineKeyboardButton(text=f"🔑 Настроить ключ {conf['name']}", callback_data=f"set_key:{active}")])
    btns.append([InlineKeyboardButton(text="⬅️ Меню", callback_data="cancel")])
    
    kb = InlineKeyboardMarkup(inline_keyboard=btns)
    
    if isinstance(event, types.Message):
        await event.answer(text=text, reply_markup=kb, parse_mode='HTML')
    elif isinstance(event, types.CallbackQuery):
        try: await event.message.edit_text(text=text, reply_markup=kb, parse_mode='HTML')
        except: pass 
        await event.answer()

@router.message(Command("settings"))
async def settings_command(m: Message):
    s = await get_user_settings(m.from_user.id)
    # Открываем вкладку текущего провайдера
    await show_tab(m, s["provider"])

@router.callback_query(F.data == "nav_main_settings")
async def settings_callback(c: types.CallbackQuery):
    s = await get_user_settings(c.from_user.id)
    await show_tab(c, s["provider"])

@router.callback_query(F.data.startswith("tab:"))
async def tab(c: types.CallbackQuery): 
    await show_tab(c, c.data.split(":")[1])

@router.callback_query(F.data.startswith("sm:"))
async def sm(c: types.CallbackQuery):
    _, p, i = c.data.split(":")
    try:
        mid = list(PROVIDERS_CONFIG[p]["models"].keys())[int(i)]
        await update_user(c.from_user.id, c.from_user.username, model=mid, provider=p)
        await c.answer(f"Выбрано: {mid} ({p})")
        await show_tab(c, p)
    except: await c.answer("Ошибка выбора")

@router.callback_query(F.data.startswith("set_key:"))
async def sk(c: types.CallbackQuery, state: FSMContext):
    p = c.data.split(":")[1]
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 Отмена", callback_data=f"tab:{p}")]])
    await c.message.edit_text(f"<a href='tg://emoji?id=5454386656628991407'>🔑</a> <b>Введите ключи для {p} (через пробел или с новой строки):</b>\nНапишите 'reset' для удаления.", reply_markup=kb, parse_mode='HTML')
    await state.update_data(kp=p)
    await state.set_state(GenStates.waiting_for_key)

@router.message(GenStates.waiting_for_key)
async def pk(m: Message, state: FSMContext):
    p = (await state.get_data())["kp"]
    
    if m.text.lower() == "reset":
        key_val = "RESET"
        count = 0
    else:
        # ИСПРАВЛЕНИЕ: Разбиваем по любым пробелам, запятым и переносам строк
        # Это решает проблему, если ключи были вставлены одной строкой
        raw_keys = re.split(r'[\s,]+', m.text.strip())
        keys = [k.strip() for k in raw_keys if k.strip()]
        
        if not keys:
            await m.answer("⚠️ Вы не прислали ни одного ключа.")
            return
        
        # Сохраняем в базу строго через \n
        key_val = "\n".join(keys)
        count = len(keys)

    args = {f"{p}_key": key_val}
    await update_user(m.from_user.id, m.from_user.username, **args)
    
    # Возвращаемся в настройки этого провайдера
    # Чтобы пользователь сразу увидел статус ключей
    s = await get_user_settings(m.from_user.id)
    
    # Если мы настроили ключи для провайдера, который НЕ активен сейчас, предупредим пользователя
    warning = ""
    if s["provider"] != p:
        warning = f"\n\n⚠️ <b>Внимание:</b> Сейчас активен провайдер <b>{s['provider']}</b>. Вы настроили <b>{p}</b>. Не забудьте выбрать модель во вкладке {p}, чтобы использовать эти ключи!"

    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 В настройки", callback_data=f"tab:{p}") ]])
    
    msg_text = "Ключи удалены." if key_val == "RESET" else f"✅ Сохранено ключей: {count} шт.{warning}"
    await m.answer(msg_text, reply_markup=kb, parse_mode='HTML')
    await state.clear()

# --- HANDLERS (GENERATION) ---

# --- 1. ФУНКЦИЯ ДЛЯ ЧТЕНИЯ ДАННЫХ ИЗ КОДА (ОБНОВЛЕННАЯ) ---
def extract_metadata(code: str, ext: str) -> Dict[str, str]:
    """Извлекает название, версию, ID и описание из кода."""
    # Значения по умолчанию
    meta = {
        "name": "GeneratedModule",
        "version": "1.0.0",
        "id": "unknown",
        "desc": "Описание отсутствует."
    }

    # Если это плагин (Extera / FTG)
    if ext == "plugin" or "__name__ =" in code:
        name_match = re.search(r'__name__\s*=\s*["\'](.*?)["\']', code)
        ver_match = re.search(r'__version__\s*=\s*["\'](.*?)["\']', code)
        id_match = re.search(r'__id__\s*=\s*["\'](.*?)["\']', code)
        desc_match = re.search(r'__description__\s*=\s*["\'](.*?)["\']', code)

        if name_match: meta["name"] = name_match.group(1)
        if ver_match: meta["version"] = ver_match.group(1)
        if id_match: meta["id"] = id_match.group(1)
        if desc_match: meta["desc"] = desc_match.group(1)

    # Если это модуль (Hikka / Heroku)
    else:
        # Ищем strings = {"name": "..."}
        hikka_name = re.search(r'strings\s*=\s*\{.*?["\']name["\']:\s*["\'](.*?)["\']', code, re.DOTALL)
        # Ищем class Name(loader.Module):
        class_name = re.search(r'class\s+(\w+)\(.*loader\.Module.*\):', code)
        # Ищем описание в """Docstring""" внутри класса
        doc_string = re.search(r'class\s+\w+\(.*loader\.Module.*\):\s*\n\s*"""(.*?)"""', code, re.DOTALL)
        
        if hikka_name:
            meta["name"] = hikka_name.group(1)
            meta["id"] = hikka_name.group(1)
        elif class_name:
            meta["name"] = class_name.group(1)
            meta["id"] = class_name.group(1)
            
        if doc_string:
            # Берем первую строку описания и убираем лишние пробелы
            meta["desc"] = doc_string.group(1).strip().split('\n')[0]
            
        # Версия в модулях редко пишется, но попробуем найти
        ver_match = re.search(r'version\s*=\s*["\'](.*?)["\']', code)
        if ver_match: meta["version"] = ver_match.group(1)

    # --- ГЕНЕРАЦИЯ БЕЗОПАСНОГО ИМЕНИ ФАЙЛА ---
    safe_name = re.sub(r'[^\w\-_\.]', '', meta["name"]).replace(" ", "")
    if not safe_name: safe_name = "result"
    meta["safe_filename"] = safe_name
    # -----------------------------------------

    # Экранируем HTML символы, чтобы не сломать разметку телеграма
    for k, v in meta.items():
        meta[k] = html.escape(str(v))
        
    return meta

# --- 2. ФУНКЦИЯ ЗАПУСКА ГЕНЕРАЦИИ (ОБНОВЛЕННАЯ) ---
async def run_gen(m: Message, state: FSMContext, sys: str, prompt: str, ext: str, is_fix=False):
    await state.set_state(GenStates.generating)
    
    # Отправляем сообщение "Генерирую..."
    wait = await m.answer("<a href='tg://emoji?id=5258281774198311547'>🧠</a> Генерирую...", parse_mode='HTML')
    
    final_prompt = prompt
    sys_prompt_final = sys
    if is_fix:
        sys_prompt_final += PROMPT_DIFF_ADDON
        
    # Делаем запрос к AI
    res = await _api_request(sys_prompt_final, final_prompt, m.from_user.id)
    
    if res.startswith("ERROR"): 
        await wait.edit_text(f"❌ {res}")
    else:
        # Получаем старый код для фикса, если нужно
        data = await state.get_data()
        original_code = data.get("original_code", "")
        
        # Применяем изменения или берем новый код
        if is_fix and original_code:
            code, note = apply_patch(original_code, res)
        else:
            code, note = apply_patch("", res)
            
        await state.update_data(original_code=code)
        
        # --- НОВАЯ ЛОГИКА ОФОРМЛЕНИЯ ---
        meta = extract_metadata(code, ext)
        
        # Формируем имя файла
        filename = f"{meta['safe_filename']}-v{meta['version']}.{ext}"
        file = BufferedInputFile(code.encode(), filename=filename)
        
        # Формируем красивую подпись
        safe_note = html.escape(note)
        
        caption_text = (
            f"📦 <b>{meta['name']}</b> v{meta['version']}\n"
            f"🆔 <code>{meta['id']}</code>\n"
            f"📄 <i>{meta['desc']}</i>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📝 <b>Ченджлог:</b>\n"
            f"<blockquote expandable>{safe_note}</blockquote>"
        )
        
        kb = [[InlineKeyboardButton(text="➕ Дополнить", callback_data=f"cont:{'mod' if ext=='py' else 'plug'}"), InlineKeyboardButton(text="🔙 Меню", callback_data="cancel")]]
        
        # Отправляем результат
        try:
            # Удаляем сообщение "Генерирую..." перед отправкой нового
            await wait.delete()
            
            if len(caption_text) > 1024:
                # Если текст длинный, шлем файл и текст отдельно
                await m.answer_document(file, caption=f"📦 <b>{meta['name']}</b>", reply_markup=InlineKeyboardMarkup(inline_keyboard=kb), parse_mode="HTML")
                await m.answer(caption_text, parse_mode="HTML")
            else:
                # Если влезает, шлем все вместе
                await m.answer_document(file, caption=caption_text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb), parse_mode="HTML")
                
        except Exception as e:
            logger.error(f"Send error: {e}")
            # Фолбэк на случай ошибки
            await m.answer_document(file, caption=f"📦 {meta['name']}\n\n📝 {safe_note[:900]}", reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))

    await state.set_state(None)

@router.callback_query(F.data == "nav_gen_mod")
async def n_gm(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5314413943035278948'>💬</a> <b>ТЗ для Heroku:</b>\nНапиши, что должен делать модуль.", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_gen_mod)

@router.message(GenStates.waiting_for_gen_mod)
async def p_gm(m: Message, state: FSMContext):
    data = await state.get_data()
    if "last_msg_id" in data:
        await safe_delete(bot, m.chat.id, data["last_msg_id"])
    await run_gen(m, state, PROMPT_HIKKA_GEN, m.text, "py", is_fix=False)

@router.callback_query(F.data == "nav_fix_mod")
async def n_fm(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5341492148468465410'>📁</a> <b>Отправь файл .py:</b>", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_fix_mod_file)

@router.message(GenStates.waiting_for_fix_mod_file, F.document)
async def p_fmf(m: Message, state: FSMContext):
    f = await bot.get_file(m.document.file_id)
    c = (await bot.download_file(f.file_path)).read().decode("utf-8", "ignore")
    await state.update_data(original_code=c)
    data = await state.get_data()
    if "last_msg_id" in data:
        await safe_delete(bot, m.chat.id, data["last_msg_id"])
    msg = await m.answer("<a href='tg://emoji?id=5465542769755826716'>✅</a> Файл принят. Что исправить?", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_fix_mod_prompt)

@router.message(GenStates.waiting_for_fix_mod_prompt)
async def p_fmp(m: Message, state: FSMContext):
    d = await state.get_data()
    original_code = d.get("original_code")
    if not original_code:
        await m.answer("❌ Нет кода. Пришли файл заново.")
        await state.set_state(GenStates.waiting_for_fix_mod_file)
        return
    if "last_msg_id" in d:
        await safe_delete(bot, m.chat.id, d["last_msg_id"])
    await run_gen(m, state, PROMPT_HIKKA_FIX, f"CODE:\n{original_code}\nREQ: {m.text}", "py", is_fix=True)

@router.callback_query(F.data == "nav_gen_plug")
async def n_gp(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5364174510708764528'>💬</a> <b>ТЗ для Extera:</b>\nОпиши функционал плагина.", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_gen_plug)

@router.message(GenStates.waiting_for_gen_plug)
async def p_gp(m: Message, state: FSMContext):
    data = await state.get_data()
    if "last_msg_id" in data:
        await safe_delete(bot, m.chat.id, data["last_msg_id"])
    await run_gen(m, state, PROMPT_EXTERA_GEN, m.text, "plugin", is_fix=False)

@router.callback_query(F.data == "nav_fix_plug")
async def n_fp(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5454419255430767770'>📁</a> <b>Отправь файл .plugin:</b>", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_fix_plug_file)

@router.message(GenStates.waiting_for_fix_plug_file, F.document)
async def handle_plugin_file(m: Message, state: FSMContext):
    if not m.document.file_name.endswith(".plugin"):
        await m.answer("❌ Это не .plugin файл.")
        return
    f = await bot.get_file(m.document.file_id)
    c = (await bot.download_file(f.file_path)).read().decode("utf-8", "ignore")
    await state.update_data(original_code=c)
    data = await state.get_data()
    if "last_msg_id" in data:
        await safe_delete(bot, m.chat.id, data["last_msg_id"])
    msg = await m.answer("<a href='tg://emoji?id=5465542769755826716'>✅</a> Файл плагина принят. Что исправить?", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_fix_plug_prompt)

@router.message(GenStates.waiting_for_fix_plug_prompt)
async def p_fpp(m: Message, state: FSMContext):
    d = await state.get_data()
    original_code = d.get("original_code")
    if not original_code:
        await m.answer("❌ Нет кода. Пришли файл заново.")
        await state.set_state(GenStates.waiting_for_fix_plug_file)
        return
    if "last_msg_id" in d:
        await safe_delete(bot, m.chat.id, d["last_msg_id"])
    await run_gen(m, state, PROMPT_EXTERA_FIX, f"CODE:\n{original_code}\nREQ: {m.text}", "plugin", is_fix=True)

@router.callback_query(F.data.startswith("cont:"))
async def cont(c: types.CallbackQuery, state: FSMContext):
    act = c.data.split(":")[1]
    await c.message.answer("📝 Что еще изменить?", reply_markup=get_cancel_kb())
    await state.set_state(GenStates.waiting_for_fix_mod_prompt if act == "mod" else GenStates.waiting_for_fix_plug_prompt)

# --- ADMIN & SYSTEM ---
@router.message(Command("admin"))
async def admin(m: Message):
    if m.from_user.id != ADMIN_ID: return
    kb = [[InlineKeyboardButton(text="📥 Скачать БД", callback_data="download_db")]]
    await m.answer("📊 Админ-панель", reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))

@router.callback_query(F.data == "download_db")
async def dl_db(c: types.CallbackQuery):
    if c.from_user.id != ADMIN_ID: return
    await c.message.answer_document(FSInputFile(DB_NAME), caption="📦 Копия базы данных")
    await c.answer()

@router.callback_query(F.data == "ignore")
async def ign(c: types.CallbackQuery): await c.answer()

async def main():
    global http_session_direct, http_session_proxy
    http_session_direct = aiohttp.ClientSession()
    if PROXY_URL:
        connector = ProxyConnector.from_url(PROXY_URL)
        http_session_proxy = aiohttp.ClientSession(connector=connector)
        print(f"Proxy connected: {PROXY_URL}")
    else:
        print("WARNING: PROXY_URL not found, using direct connection.")
        http_session_proxy = aiohttp.ClientSession()

    try:
        await init_db()
        print("Started")
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        print("🛑 Closing sessions...")
        if http_session_direct: await http_session_direct.close()
        if http_session_proxy: await http_session_proxy.close()
        print("✅ Sessions closed.")

if __name__ == "__main__":
    try: asyncio.run(main())
    except (KeyboardInterrupt, SystemExit): print("Bot stopped!")
