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
        
        # --- НОВАЯ ТАБЛИЦА ДЛЯ ОЧЕРЕДИ ЗАДАЧ ---
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
        # ---------------------------------------
        
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
    conf = PROVIDERS_CONFIG[prov]
    key = s.get(f"{prov}_key") or (os.getenv("ONLYSQ_KEY") if prov == "onlysq" else None)
    if not key: return "ERROR: Ключ не установлен в настройках."
    
    url = conf["base_url"]
    if not url.endswith("/chat/completions"):
        if url.endswith("/"): url = url[:-1]
        url = f"{url}/chat/completions"
        
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if prov == "openrouter":
        headers["HTTP-Referer"] = "https://t.me/AiModuleBot"
        headers["X-Title"] = "ModAI Bot"
        
    data = {"model": s["model"], "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}], "max_tokens": MAX_TOKENS}
    
    # --- ЛОГИКА ВЫБОРА СЕССИИ ---
    if prov == "onlysq":
        current_session = http_session_direct
    else:
        # Если прокси не задан, используем прямую, чтобы не падало
        current_session = http_session_proxy if http_session_proxy else http_session_direct

    try:
        # Используем current_session вместо http_session
        async with current_session.post(url, headers=headers, json=data, timeout=300) as resp:
            if resp.status != 200: 
                err = await resp.text()
                return f"ERROR: HTTP {resp.status} - {err[:200]}"
            res = await resp.json()
            return res["choices"][0]["message"]["content"]
    except Exception as e: return f"ERROR: {e}"

async def safe_delete(bot: Bot, chat_id: int, message_id: int):
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass # Игнорируем, если сообщение уже удалено

# --- NEW: DIFF APPLY LOGIC ---
def apply_patch(original_code: str, response_text: str) -> Tuple[str, str]:
    """
    Пытается применить SEARCH/REPLACE блоки. 
    Возвращает (итоговый код, статусное сообщение).
    """
    # 1. Сначала чистим <think>
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 2. Проверяем наличие патчей
    patch_pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>"
    matches = list(re.finditer(patch_pattern, text, re.DOTALL))
    
    if not matches:
        # Если патчей нет, ищем полный блок кода (старый режим)
        m = re.search(r"```(?:python|plugin)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            comment = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
            return code, (comment if comment else "Полная перезапись.")
        else:
            return text.strip(), "Код получен (без блоков)."

    # 3. Применяем патчи
    new_code = original_code
    applied_count = 0
    errors = []

    for match in matches:
        search_block = match.group(1) # Не стрипим, важны отступы
        replace_block = match.group(2)
        
        # Иногда LLM добавляет лишний пробел в конце SEARCH
        if search_block not in new_code:
            search_block_stripped = search_block.rstrip()
            if search_block_stripped in new_code:
                search_block = search_block_stripped
        
        if search_block in new_code:
            new_code = new_code.replace(search_block, replace_block, 1)
            applied_count += 1
        else:
            # Попытка мягкого поиска (игнорируя отступы - ОПАСНО для Python, но иногда нужно)
            # В данном случае лучше записать ошибку, чтобы не сломать код
            errors.append(f"Не найден фрагмент: {search_block[:30]}...")

    comment_text = re.sub(patch_pattern, "", text, flags=re.DOTALL).strip()
    status = f"Применено {applied_count} правок."
    if errors:
        status += f" Не найдено: {len(errors)}."
    
    return new_code, f"{status}\n\n{comment_text}"

# --- HANDLERS (START & MENU) ---

@router.message(GenStates.generating)
async def busy(m: Message): pass

@router.message(CommandStart())
async def start(m: Message, state: FSMContext):
    await update_user(m.from_user.id, m.from_user.username)
    await state.clear()
    await m.answer("<a href='tg://emoji?id=5222108309795908493'>5️⃣</a> <b>AiGen Bot</b>", reply_markup=get_main_kb(), parse_mode='HTML')

@router.callback_query(F.data == "cancel")
async def cancel(c: types.CallbackQuery, state: FSMContext):
    await state.clear()
    try:
        await c.message.edit_text("<a href='tg://emoji?id=5222108309795908493'>5️⃣</a> <b>AiGen Bot</b>", reply_markup=get_main_kb(), parse_mode='HTML')
    except Exception:
        await c.message.answer("<a href='tg://emoji?id=5222108309795908493'>5️⃣</a> <b>AiGen Bot</b>", reply_markup=get_main_kb(), parse_mode='HTML')
    await c.answer()

# --- HANDLERS (SETTINGS) ---

async def show_tab(event, active):
    user_id = event.from_user.id
    
    s = await get_user_settings(user_id)
    conf = PROVIDERS_CONFIG[active]
    
    text = f"<a href='tg://emoji?id=5301096984617166561'>5️⃣</a> <b>Выбор модели:</b>\n\n{conf['icon']} <b>{conf['name']}:</b>\n"
    for _, m in conf["models"].items(): 
        text += f"• {m['name']} — {m['desc']}\n"
    
    btns = []
    row1 = [InlineKeyboardButton(text=f"{PROVIDERS_CONFIG[p]['icon']} {PROVIDERS_CONFIG[p]['name']}", callback_data=f"tab:{p}") for p in ["onlysq", "gemini"]]
    row2 = [InlineKeyboardButton(text=f"{PROVIDERS_CONFIG[p]['icon']} {PROVIDERS_CONFIG[p]['name']}", callback_data=f"tab:{p}") for p in ["openai", "openrouter"]]
    btns.extend([row1, row2, [InlineKeyboardButton(text=f"——— {conf['icon']} {conf['name']} ———", callback_data="ignore")]])
    
    models_keys = list(conf["models"].keys())
    for i, mid in enumerate(models_keys):
        m_data = conf["models"][mid]
        mark = "✅" if (s["model"] == mid and s["provider"] == active) else m_data["icon"]
        btns.append([InlineKeyboardButton(text=f"{mark} {m_data['name']}", callback_data=f"sm:{active}:{i}")])
    
    btns.append([InlineKeyboardButton(text=f"🔑 Настроить ключ {conf['name']}", callback_data=f"set_key:{active}")])
    btns.append([InlineKeyboardButton(text="⬅️ Меню", callback_data="cancel")])
    
    kb = InlineKeyboardMarkup(inline_keyboard=btns)
    
    if isinstance(event, types.Message):
        await event.answer(text=text, reply_markup=kb, parse_mode='HTML')
    elif isinstance(event, types.CallbackQuery):
        try:
            await event.message.edit_text(text=text, reply_markup=kb, parse_mode='HTML')
        except Exception:
            pass 
        await event.answer()

@router.message(Command("settings"))
async def settings_command(m: Message):
    await show_tab(m, "onlysq")

@router.callback_query(F.data == "nav_main_settings")
async def settings_callback(c: types.CallbackQuery):
    await show_tab(c, "onlysq")

@router.callback_query(F.data.startswith("tab:"))
async def tab(c: types.CallbackQuery): 
    await show_tab(c, c.data.split(":")[1])

@router.callback_query(F.data.startswith("sm:"))
async def sm(c: types.CallbackQuery):
    _, p, i = c.data.split(":")
    try:
        mid = list(PROVIDERS_CONFIG[p]["models"].keys())[int(i)]
        await update_user(c.from_user.id, c.from_user.username, model=mid, provider=p)
        await c.answer(f"Выбрано: {mid}")
        await show_tab(c, p)
    except: await c.answer("Ошибка выбора")

@router.callback_query(F.data.startswith("set_key:"))
async def sk(c: types.CallbackQuery, state: FSMContext):
    p = c.data.split(":")[1]
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 Отмена", callback_data=f"tab:{p}")]])
    await c.message.edit_text(f"<a href='tg://emoji?id=5454386656628991407'>5️⃣</a> <b>Введите ключ для {p} (или reset):</b>", reply_markup=kb, parse_mode='HTML')
    await state.update_data(kp=p)
    await state.set_state(GenStates.waiting_for_key)

@router.message(GenStates.waiting_for_key)
async def pk(m: Message, state: FSMContext):
    p = (await state.get_data())["kp"]
    key_val = m.text.strip() if m.text.lower() != "reset" else "RESET"
    args = {f"{p}_key": key_val}
    await update_user(m.from_user.id, m.from_user.username, **args)
    
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="⚙️ Вернуться в настройки", callback_data="nav_main_settings")]])
    await m.answer("<a href='tg://emoji?id=5454079785510660283'>5️⃣</a> Сохранено.", reply_markup=kb)
    await state.clear()

# --- HANDLERS (GENERATION) ---

# In app.py
async def execute_generation(task_id, user_id, chat_id, sys, prompt, ext, is_fix, original_code, notify_msg_id=None):
    try:
        # Если это фикс, добавляем инструкции по DIFF
        final_prompt = prompt
        sys_prompt_final = sys
        if is_fix:
            sys_prompt_final += PROMPT_DIFF_ADDON
        
        # Делаем запрос (логика выбора прокси уже внутри _api_request)
        res = await _api_request(sys_prompt_final, final_prompt, user_id)
        
        if res.startswith("ERROR"):
            # Сообщаем об ошибке
            await bot.send_message(chat_id, f"❌ Ошибка генерации: {res}")
        else:
            # Применяем патч или берем код
            if is_fix and original_code:
                code, note = apply_patch(original_code, res)
            else:
                code, note = apply_patch("", res)
            
            # Подготовка файла
            file = BufferedInputFile(code.encode(), filename=f"result.{ext}")
            kb = [[InlineKeyboardButton(text="➕ Дополнить", callback_data=f"cont:{'mod' if ext=='py' else 'plug'}"), InlineKeyboardButton(text="🔙 Меню", callback_data="cancel")]]
            
            safe_note = html.escape(note)
            caption_with_quote = f"📝 Ченджлог: <blockquote expandable>{safe_note}</blockquote>"
            
            # Отправка результата
            # Используем bot.send_document, так как объекта Message может не быть (при рестарте)
            if len(caption_with_quote) > 1000:
                await bot.send_document(chat_id, file, caption="📝 Ченджлог (см. ниже):", reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))
                await bot.send_message(chat_id, caption_with_quote, parse_mode="HTML")
            else:
                await bot.send_document(chat_id, file, caption=caption_with_quote, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb), parse_mode="HTML")
            
            # Удаляем сообщение "Генерирую...", если передали ID
            if notify_msg_id:
                await safe_delete(bot, chat_id, notify_msg_id)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        await bot.send_message(chat_id, f"❌ Критическая ошибка при генерации: {e}")
    finally:
        # В ЛЮБОМ СЛУЧАЕ удаляем задачу из БД, чтобы она не зациклилась
        if task_id:
            await remove_pending_gen(task_id)

async def run_gen(m: Message, state: FSMContext, sys: str, prompt: str, ext: str, is_fix=False):
    await state.set_state(GenStates.generating)
    
    # 1. Отправляем сообщение "Генерирую..."
    wait = await m.answer("🧠 Генерирую...")
    
    # 2. Получаем original_code, если есть
    data = await state.get_data()
    original_code = data.get("original_code", "")
    
    # 3. Сохраняем задачу в БД
    task_id = await add_pending_gen(m.from_user.id, m.chat.id, sys, prompt, ext, is_fix, original_code)
    
    # 4. Запускаем выполнение в фоне (не блокируя хендлер)
    asyncio.create_task(
        execute_generation(
            task_id, m.from_user.id, m.chat.id, 
            sys, prompt, ext, is_fix, original_code, 
            notify_msg_id=wait.message_id
        )
    )
    
    # Очищаем стейт (или оставляем, как вам удобно, но original_code мы уже сохранили в БД)
    # Если нужно сохранить original_code в стейте для дальнейших правок "Дополнить", 
    # то в execute_generation нужно будет придумать, как обновить FSM, но это сложно без объекта стейта.
    # Проще всего при нажатии "Дополнить" просить скинуть файл заново или сохранять в result.py
    
    await state.set_state(None) 

@router.callback_query(F.data == "nav_gen_mod")
async def n_gm(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5314413943035278948'>5️⃣</a> <b>ТЗ для Heroku:</b>\nНапиши, что должен делать модуль.", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_gen_mod)

@router.message(GenStates.waiting_for_gen_mod)
async def p_gm(m: Message, state: FSMContext):
    # Удаляем только предыдущее сообщение бота ("Напиши ТЗ...")
    data = await state.get_data()
    if "last_msg_id" in data:
        await safe_delete(bot, m.chat.id, data["last_msg_id"])
    
    # Сообщение пользователя m.text остается в чате
    await run_gen(m, state, PROMPT_HIKKA_GEN, m.text, "py", is_fix=False)

@router.callback_query(F.data == "nav_fix_mod")
async def n_fm(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5341492148468465410'>5️⃣</a> <b>Отправь файл .py:</b>", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_fix_mod_file)

@router.message(GenStates.waiting_for_fix_mod_file, F.document)
async def p_fmf(m: Message, state: FSMContext):
    f = await bot.get_file(m.document.file_id)
    c = (await bot.download_file(f.file_path)).read().decode("utf-8", "ignore")
    await state.update_data(original_code=c)
    
    # Удаляем просьбу "Отправь файл"
    data = await state.get_data()
    if "last_msg_id" in data:
        await safe_delete(bot, m.chat.id, data["last_msg_id"])
        
    msg = await m.answer("<a href='tg://emoji?id=5465542769755826716'>5️⃣</a> Файл принят. Что исправить?", reply_markup=get_cancel_kb())
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_fix_mod_prompt)

@router.message(GenStates.waiting_for_fix_mod_prompt)
async def p_fmp(m: Message, state: FSMContext):
    d = await state.get_data()
    
    # Удаляем вопрос "Что исправить?"
    if "last_msg_id" in d:
        await safe_delete(bot, m.chat.id, d["last_msg_id"])
    
    # Сообщение пользователя с просьбой фикса остается
    await run_gen(m, state, PROMPT_HIKKA_FIX, f"CODE:\n{d['original_code']}\nREQ: {m.text}", "py", is_fix=True)

@router.callback_query(F.data == "nav_gen_plug")
async def n_gp(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5364174510708764528'>5️⃣</a> <b>ТЗ для Extera:</b>\nОпиши функционал плагина.", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_gen_plug)

@router.message(GenStates.waiting_for_gen_plug)
async def p_gp(m: Message, state: FSMContext):
    data = await state.get_data()
    # Удаляем вопрос бота
    if "last_msg_id" in data:
        await safe_delete(bot, m.chat.id, data["last_msg_id"])
        
    await run_gen(m, state, PROMPT_EXTERA_GEN, m.text, "plugin", is_fix=False)

@router.callback_query(F.data == "nav_fix_plug")
async def n_fp(c: types.CallbackQuery, state: FSMContext):
    msg = await c.message.edit_text("<a href='tg://emoji?id=5454419255430767770'>5️⃣</a> <b>Отправь файл .plugin:</b>", reply_markup=get_cancel_kb(), parse_mode='HTML')
    await state.update_data(last_msg_id=msg.message_id)
    await state.set_state(GenStates.waiting_for_fix_plug_file)

@router.message(GenStates.waiting_for_fix_plug_file, F.document)
async def handle_plugin_file(message: types.Message, state: FSMContext):
    # Проверка расширения файла
    if message.document.file_name.endswith(".plugin"):
        # Твоя логика обработки файла
        await message.answer("<a href='tg://emoji?id=5219899949281453881'>5️⃣</a> Файл получен")
        await state.clear()
    else:
        await message.answer("<a href='tg://emoji?id=5454225015534805938'>5️⃣</a> Это не .plugin файл. Попробуй еще раз.")

@router.message(GenStates.waiting_for_fix_plug_prompt)
async def p_fpp(m: Message, state: FSMContext):
    d = await state.get_data()
    
    # Удаляем "Что исправить?"
    if "last_msg_id" in d:
        await safe_delete(bot, m.chat.id, d["last_msg_id"])
        
    await run_gen(m, state, PROMPT_EXTERA_FIX, f"CODE:\n{d['original_code']}\nREQ: {m.text}", "plugin", is_fix=True)

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

async def restore_pending_generations():
    tasks = await get_all_pending_gens()
    if not tasks:
        return
    
    print(f"🔄 Восстановление {len(tasks)} прерванных генераций...")
    
    for task in tasks:
        # task - это строка из БД (Row object)
        # Уведомляем пользователя, что мы не забыли про него
        try:
            await bot.send_message(task['chat_id'], "🔄 Бот был перезагружен. Возобновляю вашу генерацию...")
        except:
            pass # Если юзер заблокировал бота, просто работаем дальше

        # Запускаем задачу
        asyncio.create_task(
            execute_generation(
                task_id=task['id'],
                user_id=task['user_id'],
                chat_id=task['chat_id'],
                sys=task['sys_prompt'],
                prompt=task['user_prompt'],
                ext=task['ext'],
                is_fix=bool(task['is_fix']),
                original_code=task['original_code'],
                notify_msg_id=None # Старое сообщение "Генерирую" мы уже не найдем/не удалим
            )
        )

async def main():
    global http_session_direct, http_session_proxy
    
    # --- 1. Создаем сессии ---
    http_session_direct = aiohttp.ClientSession()

    if PROXY_URL:
        # Подключаем прокси (SOCKS4/5)
        connector = ProxyConnector.from_url(PROXY_URL)
        http_session_proxy = aiohttp.ClientSession(connector=connector)
        print(f"Proxy connected: {PROXY_URL}")
    else:
        print("WARNING: PROXY_URL not found, using direct connection.")
        http_session_proxy = aiohttp.ClientSession()

    # --- 2. Запускаем бота в блоке try ---
    try:
        await init_db()
        
        # Если вы уже добавили функцию восстановления задач из прошлого ответа:
        # await restore_pending_generations() 
        
        print("Started")
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
        
    # --- 3. Этот блок выполнится ВСЕГДА при остановке бота ---
    finally:
        print("🛑 Closing sessions...")
        if http_session_direct:
            await http_session_direct.close()
        if http_session_proxy:
            await http_session_proxy.close()
        print("✅ Sessions closed.")

if __name__ == "__main__":
    try: 
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped!")

if __name__ == "__main__":
    try: asyncio.run(main())
    except: pass
