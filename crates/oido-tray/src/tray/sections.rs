//! Trait `MenuSection` — abstracción declarativa sobre el menú nativo
//! de bandeja. Cada sección del menú (hotkey, tema, modo, idioma,
//! prompt, modelos, exit) implementa este trait y se registra en
//! `PlatformTray::with_sections`.
//!
//! ## i18n
//!
//! Las secciones NO almacenan strings propios. Cada una guarda el
//! `UiLanguage` activo y, en `build()`, consulta
//! [`crate::tray::i18n::strings`] para obtener el `&'static Strings`
//! correspondiente. Esto evita duplicación, garantiza que cambiar
//! `UiLanguage` se refleja en el próximo `rebuild_menu`, y mantiene
//! el contrato de `MenuSection: 'static` (las secciones no tienen
//! lifetimes).
//!
//! ## Compatibilidad
//!
//! El árbol visible para el usuario es esencialmente idéntico al
//! anterior; las 2 secciones nuevas (Idioma, Prompt) se insertan
//! antes de "Modelos" para mantener la agrupación de "preferencias
//! del usuario" en la parte superior.

use std::path::PathBuf;

use oido_config::{EffortPreset, PromptPreset, SttMode, Theme, UiLanguage};
use oido_models::{ModelEntry, ModelFamily};

use crate::traits::MenuAction;
use crate::tray::i18n::{strings, Strings};

/// Prefijo canónico para los items del submenú Micrófono. El bin
/// extrae el nombre del dispositivo con `id.strip_prefix("mic:")` en
/// `id_to_action` y lo pasa a `MenuAction::SetInputDevice(name)`.
const MIC_ITEM_PREFIX: &str = "mic:";

/// ID canónico de un item, estable entre OS. Sirve como clave del
/// `HashMap<MenuId, MenuAction>` que el forwarder necesita para mapear
/// clics a acciones.
///
/// Es `String` (no `&'static str`) para soportar ids dinámicos como
/// `"model:ggml-base.bin"` que se generan a partir del catálogo de
/// modelos. Los ids estáticos (cambiar hotkey, abrir carpeta, etc.) se
/// pasan como `String::from("…")`.
pub type ItemId = String;

/// Descripción agnóstica de un item de menú.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuItemSpec {
    pub id: ItemId,
    pub label: String,
    pub enabled: bool,
}

/// Descripción de un sub-árbol de menú.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Section {
    /// Item simple en el nivel actual.
    Item(MenuItemSpec),
    /// Submenú con título e items anidados.
    Submenu {
        label: String,
        items: Vec<MenuItemSpec>,
    },
    /// Separador visual. No tiene items.
    Separator,
}

/// Sección declarativa. Implementar este trait permite que el backend
/// de bandeja la convierta a items nativos (tray-icon/ksni) sin que
/// la sección conozca el backend.
pub trait MenuSection: Send + Sync + 'static {
    /// Identificador único (útil para diagnóstico/tests). No se usa
    /// para mapear clics; ese mapeo va por `MenuItemSpec::id`.
    fn id(&self) -> &'static str;

    /// Construye la lista de items / submenús que componen esta
    /// sección. El backend se encarga de añadir separadores antes y
    /// después según las convenciones del OS.
    fn build(&self) -> Vec<Section>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Marca ✓ o vacío según `active`. Replica el patrón visual previo
/// pero aplicado a subsecciones (idioma, prompt). El espacio al
/// principio mantiene la alineación con los items sin marca.
fn check_or_blank(active: bool) -> &'static str {
    if active {
        "✓ "
    } else {
        "  "
    }
}

/// Construye el `id` canónico del item a partir del filename del modelo.
/// Formato: `"model:{filename}"`. Lo parsea `id_to_action` y el bin lo
/// extrae del `MenuAction::ModelItem(String)`.
pub fn item_id(filename: &str) -> ItemId {
    format!("model:{filename}")
}

// ---------------------------------------------------------------------------
// Secciones existentes — refactor a `UiLanguage` interno
// ---------------------------------------------------------------------------

/// Sección de aviso no modal: visible SOLO cuando hay mismatch entre un
/// modelo solo-inglés activo y un idioma distinto a "en". Renderiza un
/// único item raíz con id `"fix_model_lang"` que, al pulsarse, dispara
/// `MenuAction::FixModelLanguage`. El bin, en el handler, recalcula el
/// contraparte multilingüe desde la config (`oido_models::multilingual_
/// counterpart`) y delega en el mismo `handle_model_click` que el
/// submenú Modelos (descarga si no está, activa si está).
///
/// Llevamos el `suggested_filename` solo para componer el label del
/// item ("⚠ Cambiar a modelo multilingüe (ggml-small.bin)…"). El `id`
/// canónico es siempre `"fix_model_lang"` para que `id_to_action` no
/// tenga que parsear nombres de archivo.
#[derive(Debug)]
pub struct ModelLangMismatchSection {
    pub ui_language: UiLanguage,
    pub suggested_filename: Option<String>,
}

impl MenuSection for ModelLangMismatchSection {
    fn id(&self) -> &'static str {
        "model_lang_mismatch"
    }
    fn build(&self) -> Vec<Section> {
        // Sin sugerencia → no renderizamos nada. Esta rama es la que
        // mantiene el árbol idéntico al previo cuando NO hay mismatch.
        let Some(suggested) = self.suggested_filename.as_deref() else {
            return Vec::new();
        };
        let s = strings(self.ui_language);
        let label = format!("{} ({})", s.model_lang_mismatch_action, suggested);
        vec![Section::Item(MenuItemSpec {
            id: "fix_model_lang".into(),
            label,
            enabled: true,
        })]
    }
}

/// Sección "Cambiar hotkey". Item simple en el nivel raíz.
#[derive(Debug)]
pub struct HotkeySection {
    pub ui_language: UiLanguage,
}

impl MenuSection for HotkeySection {
    fn id(&self) -> &'static str {
        "hotkey"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        vec![Section::Item(MenuItemSpec {
            id: "change_hotkey".into(),
            label: s.change_hotkey.into(),
            enabled: true,
        })]
    }
}

/// Sección "Tema" — submenú con 3 opciones de tema.
#[derive(Debug)]
pub struct ThemeSection {
    pub ui_language: UiLanguage,
    pub current: Theme,
}

impl MenuSection for ThemeSection {
    fn id(&self) -> &'static str {
        "theme"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        vec![Section::Submenu {
            label: s.theme.into(),
            items: vec![
                MenuItemSpec {
                    id: "theme_dark".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == Theme::Dark),
                        s.theme_dark
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "theme_light".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == Theme::Light),
                        s.theme_light
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "theme_system".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == Theme::System),
                        s.theme_system
                    ),
                    enabled: true,
                },
            ],
        }]
    }
}

/// Sección "Modo de dictado" — submenú con Batch / Streaming / Chunked.
#[derive(Debug)]
pub struct ModeSection {
    pub ui_language: UiLanguage,
    pub current: SttMode,
}

impl MenuSection for ModeSection {
    fn id(&self) -> &'static str {
        "mode"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        vec![Section::Submenu {
            label: s.mode.into(),
            items: vec![
                MenuItemSpec {
                    id: "mode_batch".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == SttMode::Batch),
                        s.mode_batch
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "mode_streaming".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == SttMode::Streaming),
                        s.mode_streaming
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "mode_chunked".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == SttMode::Chunked),
                        s.mode_chunked
                    ),
                    enabled: true,
                },
            ],
        }]
    }
}

// ---------------------------------------------------------------------------
// Secciones NUEVAS: Idioma de la UI + Prompt del sistema + Esfuerzo
// ---------------------------------------------------------------------------

/// Sección "Esfuerzo de decodificación" — submenú con los 3 presets
/// que controlan los parámetros de `FullParams` de whisper.cpp
/// (estrategia de muestreo, `temperature_inc`, `entropy_thold`,
/// `length_penalty`).
///
/// Ver `oido_stt::preset_settings` para el mapping concreto de cada
/// preset a los setters de `FullParams`.
#[derive(Debug)]
pub struct EffortSection {
    pub ui_language: UiLanguage,
    pub current: EffortPreset,
}

impl MenuSection for EffortSection {
    fn id(&self) -> &'static str {
        "effort"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        vec![Section::Submenu {
            label: s.effort.into(),
            items: vec![
                MenuItemSpec {
                    id: "effort_balanced".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == EffortPreset::Balanced),
                        s.effort_balanced
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "effort_robust".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == EffortPreset::Robust),
                        s.effort_robust
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "effort_high_quality".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == EffortPreset::HighQuality),
                        s.effort_high_quality
                    ),
                    enabled: true,
                },
            ],
        }]
    }
}

/// Sección "Idioma de la interfaz" — submenú con ES / EN / Bilingüe.
#[derive(Debug)]
pub struct UiLanguageSection {
    pub current: UiLanguage,
}

impl MenuSection for UiLanguageSection {
    fn id(&self) -> &'static str {
        "ui_language"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.current);
        vec![Section::Submenu {
            label: s.ui_language.into(),
            items: vec![
                MenuItemSpec {
                    id: "ui_es".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == UiLanguage::Es),
                        s.ui_es
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "ui_en".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == UiLanguage::En),
                        s.ui_en
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "ui_bil".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == UiLanguage::Bilingual),
                        s.ui_bilingual
                    ),
                    enabled: true,
                },
            ],
        }]
    }
}

/// Sección "Prompt del sistema" — submenú con los presets y un atajo a
/// la edición del config.json (campo `system_prompt`).
///
/// - `prompt_bilingual` / `prompt_es` / `prompt_en`: presets
///   hardcoded sin texto editable.
/// - `prompt_custom`: selector del preset Custom. El label lleva un
///   preview (o un hint apuntando a la CLI `--set-prompt`).
/// - `prompt_edit`: abre el `config.json` del usuario en el editor de
///   texto nativo del OS (Bloc de Notas en Windows, xdg-open en
///   Linux, `open -t` en macOS). Es la única vía para editar el campo
///   `system_prompt` desde la interfaz sin tirar de CLI.
#[derive(Debug)]
pub struct PromptSection {
    pub ui_language: UiLanguage,
    pub current: PromptPreset,
    pub custom_text: String, // contenido actual del system_prompt (vacío si no hay)
}

impl MenuSection for PromptSection {
    fn id(&self) -> &'static str {
        "prompt"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        let custom_label = if self.custom_text.is_empty() {
            format!(
                "{}{}  {}",
                check_or_blank(self.current == PromptPreset::Custom),
                s.prompt_custom,
                s.prompt_custom_hint
            )
        } else {
            // Truncar preview a 40 chars para no hacer el item enorme.
            let preview: String = self.custom_text.chars().take(40).collect();
            let ellipsis = if self.custom_text.chars().count() > 40 {
                "…"
            } else {
                ""
            };
            format!(
                "{}{}  \"{}{}\"",
                check_or_blank(self.current == PromptPreset::Custom),
                s.prompt_custom,
                preview,
                ellipsis
            )
        };
        vec![Section::Submenu {
            label: s.system_prompt.into(),
            items: vec![
                MenuItemSpec {
                    id: "prompt_bilingual".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == PromptPreset::BilingualEsEn),
                        s.prompt_bilingual
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "prompt_es".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == PromptPreset::SpanishOnly),
                        s.prompt_es
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "prompt_en".into(),
                    label: format!(
                        "{}{}",
                        check_or_blank(self.current == PromptPreset::EnglishOnly),
                        s.prompt_en
                    ),
                    enabled: true,
                },
                MenuItemSpec {
                    id: "prompt_custom".into(),
                    label: custom_label,
                    enabled: true,
                },
                MenuItemSpec {
                    id: "prompt_edit".into(),
                    label: s.prompt_edit.into(),
                    enabled: true,
                },
            ],
        }]
    }
}

// ---------------------------------------------------------------------------
// Sección "Micrófono" — submenú dinámico con los dispositivos de
// entrada detectados por cpal, más un item "Automático" (default del
// OS) y un disparador de re-sondeo de calidad.
// ---------------------------------------------------------------------------

/// Dispositivo de entrada, en el formato mínimo que necesita la
/// sección. Lo construye el bin desde `oido_audio::list_input_devices()`
/// y lo inyecta en `BuildContext` para mantener este crate sin
/// dependencias de cpal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicDevice {
    pub name: String,
    pub is_default: bool,
}

#[derive(Debug)]
pub struct MicrophoneSection {
    pub devices: Vec<MicDevice>,
    /// `None` = modo automático (default del OS). `Some(name)` = fijado
    /// al dispositivo con ese nombre exacto.
    pub active: Option<String>,
    pub ui_language: UiLanguage,
}

impl MicrophoneSection {
    pub fn new(devices: Vec<MicDevice>, active: Option<String>, ui_language: UiLanguage) -> Self {
        Self {
            devices,
            active,
            ui_language,
        }
    }
}

/// Sufijo corto "← activo" / "← active" para el item "Automático" del
/// submenú Micrófono. Se localiza por heurística sobre el label del
/// submenú (los 3 idiomas los distinguimos así sin añadir un campo
/// nuevo a `Strings`).
fn mic_active_short(ui_language: UiLanguage) -> &'static str {
    match ui_language {
        UiLanguage::En => "  ← active",
        _ => "  ← activo",
    }
}

impl MenuSection for MicrophoneSection {
    fn id(&self) -> &'static str {
        "microphone"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        let mut items: Vec<MenuItemSpec> = Vec::new();

        // Item "Automático" — siempre primero, ✓ cuando active.is_none().
        let auto_active = self.active.is_none();
        let mut auto_label = format!("{}{}", check_or_blank(auto_active), s.mic_auto);
        if auto_active {
            auto_label.push_str(mic_active_short(self.ui_language));
        }
        items.push(MenuItemSpec {
            id: "mic_auto".into(),
            label: auto_label,
            enabled: true,
        });

        if self.devices.is_empty() {
            items.push(MenuItemSpec {
                id: String::new(),
                label: format!("  {}", s.mic_none),
                enabled: false,
            });
        } else {
            for d in &self.devices {
                let is_active = self.active.as_deref() == Some(d.name.as_str());
                let mut label = format!("{}{}", check_or_blank(is_active), d.name);
                if d.is_default {
                    label.push_str("  (default)");
                }
                if is_active {
                    label.push_str(s.mic_active);
                }
                items.push(MenuItemSpec {
                    id: format!("{MIC_ITEM_PREFIX}{}", d.name),
                    label,
                    enabled: true,
                });
            }
        }

        // Separador visual + item de re-sondeo (lo aísla del listado
        // de dispositivos, igual que `open_models_dir` y `check_updates`
        // en la sección Modelos).
        items.push(MenuItemSpec {
            id: String::new(),
            label: "─────────────────".into(),
            enabled: false,
        });
        items.push(MenuItemSpec {
            id: "mic_reprobe".into(),
            label: s.mic_reprobe.into(),
            enabled: true,
        });

        vec![Section::Submenu {
            label: s.microphone.into(),
            items,
        }]
    }
}

// ---------------------------------------------------------------------------
// Sección "Modelos" — refactor para usar strings i18n
// ---------------------------------------------------------------------------

/// Sección "Modelos" — submenú dinámico con cada entry del catálogo de
/// `oido_models`. Cada item se marca con `✓` (instalado) o `↓ Descargar`
/// (no instalado) según el contenido de `models_dir` al construir la
/// sección. El item activo lleva el sufijo `← activo`.
///
/// Modelos `.en` (solo inglés) llevan un sufijo `⚠ Solo inglés` para
/// que el usuario sepa que NO entenderá español, sin ocultar la opción.
#[derive(Debug)]
pub struct ModelsSection {
    pub models_dir: PathBuf,
    pub active_model: String,
    pub ui_language: UiLanguage,
}

impl ModelsSection {
    pub fn new(
        models_dir: PathBuf,
        active_model: impl Into<String>,
        ui_language: UiLanguage,
    ) -> Self {
        Self {
            models_dir,
            active_model: active_model.into(),
            ui_language,
        }
    }

    /// Renderiza un único item del submenú a partir de un entry del catálogo.
    fn render_item(
        &self,
        s: &Strings,
        entry: &ModelEntry,
        installed: bool,
        active: bool,
    ) -> MenuItemSpec {
        let state = if installed {
            s.model_installed
        } else {
            s.model_download
        };
        // Para modelos .en (solo inglés) añadimos un sufijo de aviso
        // visual. NO bloqueamos la activación — el usuario decide.
        let is_en_only = entry.filename.ends_with(".en.bin");
        let en_warning = if is_en_only { s.model_en_only } else { "" };
        let mut label = format!(
            "{}{} ({}){}",
            state,
            entry.filename,
            oido_models::human_size(entry.size_bytes),
            en_warning
        );
        if active {
            label.push_str(s.model_active);
        }
        MenuItemSpec {
            id: item_id(&entry.filename),
            label,
            enabled: true,
        }
    }
}

impl MenuSection for ModelsSection {
    fn id(&self) -> &'static str {
        "models"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        let installed = oido_models::list_installed(&self.models_dir).unwrap_or_default();
        let mut model_items: Vec<MenuItemSpec> = Vec::new();
        for family in [
            ModelFamily::Tiny,
            ModelFamily::Base,
            ModelFamily::Small,
            ModelFamily::Medium,
            ModelFamily::Large,
            ModelFamily::Vad,
        ] {
            // Encabezado de familia (item deshabilitado para que actúe
            // como separador visual).
            let family_label: &'static str = match family {
                ModelFamily::Tiny => s.model_tiny,
                ModelFamily::Base => s.model_base,
                ModelFamily::Small => s.model_small,
                ModelFamily::Medium => s.model_medium,
                ModelFamily::Large => s.model_large,
                ModelFamily::Vad => s.model_vad,
            };
            model_items.push(MenuItemSpec {
                id: String::new(),
                label: format!("── {family_label} ──"),
                enabled: false,
            });
            for entry in oido_models::catalog().iter().filter(|e| e.family == family) {
                let installed = installed.iter().any(|f| f == &entry.filename);
                let active = entry.filename == self.active_model;
                model_items.push(self.render_item(s, entry, installed, active));
            }
        }
        vec![
            Section::Submenu {
                label: s.models.into(),
                items: model_items,
            },
            Section::Item(MenuItemSpec {
                id: "open_models_dir".into(),
                label: s.open_models_dir.into(),
                enabled: true,
            }),
            Section::Item(MenuItemSpec {
                id: "check_updates".into(),
                label: s.check_updates.into(),
                enabled: true,
            }),
        ]
    }
}

// ---------------------------------------------------------------------------
// Sección "Salir"
// ---------------------------------------------------------------------------

/// Sección "Salir" — siempre la última; item simple.
#[derive(Debug)]
pub struct ExitSection {
    pub ui_language: UiLanguage,
}

impl MenuSection for ExitSection {
    fn id(&self) -> &'static str {
        "exit"
    }
    fn build(&self) -> Vec<Section> {
        let s = strings(self.ui_language);
        vec![Section::Item(MenuItemSpec {
            id: "exit".into(),
            label: s.exit.into(),
            enabled: true,
        })]
    }
}

// ---------------------------------------------------------------------------
// default_sections — composición canónica
// ---------------------------------------------------------------------------

/// Contexto necesario para construir el árbol de menú. Se pasa por
/// referencia a `default_sections` para evitar listas largas de
/// parámetros y para que un futuro campo extra (p.ej. `prompt_preset`
/// cuando se añada más state) no rompa la API.
///
/// Diseño: este struct es la **fuente única de verdad** sobre el
/// estado que el menú refleja. El bin debe mantenerlo en sincronía
/// con `Config` (cada `replace + save` debe ir seguido de un
/// `rebuild_menu` con un `BuildContext` actualizado).
#[derive(Debug, Clone)]
pub struct BuildContext {
    pub models_dir: PathBuf,
    pub active_model: String,
    pub ui_language: UiLanguage,
    pub theme: Theme,
    pub stt_mode: SttMode,
    pub prompt_preset: PromptPreset,
    pub prompt_custom_text: String,
    pub effort: EffortPreset,
    /// `Some(filename)` cuando hay mismatch entre un modelo solo-inglés
    /// activo y un idioma distinto a "en"; el bin pasa el filename del
    /// modelo multilingüe equivalente (p.ej. `"ggml-small.bin"`). Si es
    /// `None`, no se renderiza la sección de aviso.
    pub model_lang_mismatch: Option<String>,
    /// Dispositivos de entrada detectados por cpal, en el formato
    /// ligero que `MicrophoneSection` espera (sin tipos cpal en la API
    /// pública de este crate). El bin la construye con
    /// `oido_audio::list_input_devices()` en cada `build_ctx`.
    pub input_devices: Vec<MicDevice>,
    /// `None` = modo automático (default del OS). `Some(name)` =
    /// dispositivo fijado por el usuario. Lo más reciente de
    /// `Config::input_device`.
    pub input_device: Option<String>,
}

impl BuildContext {
    /// Constructor de conveniencia con defaults razonables para el
    /// primer render (antes de tener `Config` materializada). El bin
    /// debe reemplazarlo con un contexto real tras el startup.
    #[must_use]
    pub fn initial(models_dir: PathBuf, active_model: impl Into<String>) -> Self {
        Self {
            models_dir,
            active_model: active_model.into(),
            ui_language: UiLanguage::Es,
            theme: Theme::System,
            stt_mode: SttMode::Batch,
            prompt_preset: PromptPreset::BilingualEsEn,
            prompt_custom_text: String::new(),
            effort: EffortPreset::Balanced,
            model_lang_mismatch: None,
            input_devices: Vec::new(),
            input_device: None,
        }
    }
}

/// Conjunto canónico de secciones.
///
/// El bin llama a `Tray::rebuild_menu` después de cada descarga o
/// cambio de preferencia para refrescar el árbol.
pub fn default_sections(ctx: &BuildContext) -> Vec<Box<dyn MenuSection>> {
    let mut out: Vec<Box<dyn MenuSection>> = Vec::with_capacity(8);
    // Aviso de mismatch: primero para máxima visibilidad. Solo se añade
    // cuando hay sugerencia (Some); con None la sección renderiza 0 items
    // pero igual la dejamos fuera para mantener `len() == 7` en el caso
    // normal (consistente con tests previos).
    if ctx.model_lang_mismatch.is_some() {
        out.push(Box::new(ModelLangMismatchSection {
            ui_language: ctx.ui_language,
            suggested_filename: ctx.model_lang_mismatch.clone(),
        }));
    }
    out.push(Box::new(HotkeySection {
        ui_language: ctx.ui_language,
    }));
    out.push(Box::new(ThemeSection {
        ui_language: ctx.ui_language,
        current: ctx.theme,
    }));
    out.push(Box::new(ModeSection {
        ui_language: ctx.ui_language,
        current: ctx.stt_mode,
    }));
    out.push(Box::new(EffortSection {
        ui_language: ctx.ui_language,
        current: ctx.effort,
    }));
    out.push(Box::new(UiLanguageSection {
        current: ctx.ui_language,
    }));
    out.push(Box::new(PromptSection {
        ui_language: ctx.ui_language,
        current: ctx.prompt_preset,
        custom_text: ctx.prompt_custom_text.clone(),
    }));
    out.push(Box::new(MicrophoneSection::new(
        ctx.input_devices.clone(),
        ctx.input_device.clone(),
        ctx.ui_language,
    )));
    out.push(Box::new(ModelsSection::new(
        ctx.models_dir.clone(),
        ctx.active_model.clone(),
        ctx.ui_language,
    )));
    out.push(Box::new(ExitSection {
        ui_language: ctx.ui_language,
    }));
    out
}

// ---------------------------------------------------------------------------
// id_to_action — mapeo canónico
// ---------------------------------------------------------------------------

/// Mapea el `id` canónico de un `MenuItemSpec` a la acción que debe
/// disparar el menú. Centraliza la traducción para que el forwarder
/// no necesite un map gigante con un `if` por item.
pub fn id_to_action(id: &str) -> Option<MenuAction> {
    Some(match id {
        "change_hotkey" => MenuAction::ChangeHotkey,
        "theme_dark" => MenuAction::SetTheme(Theme::Dark),
        "theme_light" => MenuAction::SetTheme(Theme::Light),
        "theme_system" => MenuAction::SetTheme(Theme::System),
        "mode_batch" => MenuAction::SetSttMode(SttMode::Batch),
        "mode_streaming" => MenuAction::SetSttMode(SttMode::Streaming),
        "mode_chunked" => MenuAction::SetSttMode(SttMode::Chunked),
        "effort_balanced" => MenuAction::SetEffort(EffortPreset::Balanced),
        "effort_robust" => MenuAction::SetEffort(EffortPreset::Robust),
        "effort_high_quality" => MenuAction::SetEffort(EffortPreset::HighQuality),
        "ui_es" => MenuAction::SetUiLanguage(UiLanguage::Es),
        "ui_en" => MenuAction::SetUiLanguage(UiLanguage::En),
        "ui_bil" => MenuAction::SetUiLanguage(UiLanguage::Bilingual),
        "prompt_bilingual" => MenuAction::SetPromptPreset(PromptPreset::BilingualEsEn),
        "prompt_es" => MenuAction::SetPromptPreset(PromptPreset::SpanishOnly),
        "prompt_en" => MenuAction::SetPromptPreset(PromptPreset::EnglishOnly),
        "prompt_custom" => MenuAction::SetPromptPreset(PromptPreset::Custom),
        "prompt_edit" => MenuAction::EditPrompt,
        "open_models_dir" => MenuAction::OpenModelsDir,
        "check_updates" => MenuAction::CheckUpdates,
        "exit" => MenuAction::Exit,
        "fix_model_lang" => MenuAction::FixModelLanguage(String::new()),
        "mic_auto" => MenuAction::SetInputDevice(String::new()),
        "mic_reprobe" => MenuAction::ProbeMicrophones,
        id if id.starts_with("mic:") => {
            MenuAction::SetInputDevice(id[MIC_ITEM_PREFIX.len()..].to_string())
        }
        id if id.starts_with("model:") => MenuAction::ModelItem(id["model:".len()..].to_string()),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn id_to_action_covers_known_ids() {
        assert_eq!(
            id_to_action("change_hotkey"),
            Some(MenuAction::ChangeHotkey)
        );
        assert_eq!(
            id_to_action("theme_dark"),
            Some(MenuAction::SetTheme(Theme::Dark))
        );
        assert_eq!(
            id_to_action("theme_light"),
            Some(MenuAction::SetTheme(Theme::Light))
        );
        assert_eq!(
            id_to_action("theme_system"),
            Some(MenuAction::SetTheme(Theme::System))
        );
        assert_eq!(
            id_to_action("mode_batch"),
            Some(MenuAction::SetSttMode(SttMode::Batch))
        );
        assert_eq!(
            id_to_action("mode_streaming"),
            Some(MenuAction::SetSttMode(SttMode::Streaming))
        );
        assert_eq!(
            id_to_action("mode_chunked"),
            Some(MenuAction::SetSttMode(SttMode::Chunked))
        );
        // effort
        assert_eq!(
            id_to_action("effort_balanced"),
            Some(MenuAction::SetEffort(EffortPreset::Balanced))
        );
        assert_eq!(
            id_to_action("effort_robust"),
            Some(MenuAction::SetEffort(EffortPreset::Robust))
        );
        assert_eq!(
            id_to_action("effort_high_quality"),
            Some(MenuAction::SetEffort(EffortPreset::HighQuality))
        );
        // i18n
        assert_eq!(
            id_to_action("ui_es"),
            Some(MenuAction::SetUiLanguage(UiLanguage::Es))
        );
        assert_eq!(
            id_to_action("ui_en"),
            Some(MenuAction::SetUiLanguage(UiLanguage::En))
        );
        assert_eq!(
            id_to_action("ui_bil"),
            Some(MenuAction::SetUiLanguage(UiLanguage::Bilingual))
        );
        // prompt
        assert_eq!(
            id_to_action("prompt_bilingual"),
            Some(MenuAction::SetPromptPreset(PromptPreset::BilingualEsEn))
        );
        assert_eq!(
            id_to_action("prompt_es"),
            Some(MenuAction::SetPromptPreset(PromptPreset::SpanishOnly))
        );
        assert_eq!(
            id_to_action("prompt_en"),
            Some(MenuAction::SetPromptPreset(PromptPreset::EnglishOnly))
        );
        assert_eq!(
            id_to_action("prompt_custom"),
            Some(MenuAction::SetPromptPreset(PromptPreset::Custom))
        );
        assert_eq!(id_to_action("prompt_edit"), Some(MenuAction::EditPrompt));
        assert_eq!(
            id_to_action("open_models_dir"),
            Some(MenuAction::OpenModelsDir)
        );
        assert_eq!(
            id_to_action("check_updates"),
            Some(MenuAction::CheckUpdates)
        );
        assert_eq!(id_to_action("exit"), Some(MenuAction::Exit));
        assert_eq!(
            id_to_action("model:ggml-base.bin"),
            Some(MenuAction::ModelItem("ggml-base.bin".to_string()))
        );
        // microphone
        assert_eq!(
            id_to_action("mic_auto"),
            Some(MenuAction::SetInputDevice(String::new()))
        );
        assert_eq!(
            id_to_action("mic_reprobe"),
            Some(MenuAction::ProbeMicrophones)
        );
        assert_eq!(
            id_to_action("mic:USB Microphone"),
            Some(MenuAction::SetInputDevice("USB Microphone".to_string()))
        );
    }

    #[test]
    fn id_to_action_returns_none_for_unknown() {
        assert_eq!(id_to_action("nonexistent_item"), None);
        assert_eq!(id_to_action(""), None);
    }

    fn empty_models_dir() -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        (dir, path)
    }

    fn call_default(dir: PathBuf) -> Vec<Box<dyn MenuSection>> {
        let ctx = BuildContext {
            models_dir: dir,
            active_model: "ggml-base.bin".into(),
            ui_language: UiLanguage::Es,
            theme: Theme::System,
            stt_mode: SttMode::Batch,
            prompt_preset: PromptPreset::BilingualEsEn,
            prompt_custom_text: String::new(),
            effort: EffortPreset::Balanced,
            model_lang_mismatch: None,
            input_devices: Vec::new(),
            input_device: None,
        };
        default_sections(&ctx)
    }

    #[test]
    fn default_sections_produces_expected_tree() {
        let (_tmp, dir) = empty_models_dir();
        let sections = call_default(dir);
        assert_eq!(
            sections.len(),
            9,
            "9 secciones: hotkey, theme, mode, effort, ui_language, prompt, microphone, models, exit"
        );
        for s in &sections {
            assert!(
                !s.build().is_empty(),
                "sección {} no debe estar vacía",
                s.id()
            );
        }
    }

    #[test]
    fn default_sections_models_submenu_marks_installed_and_active() {
        let (_tmp, dir) = empty_models_dir();
        std::fs::write(dir.join("ggml-tiny.bin"), b"x").unwrap();
        std::fs::write(dir.join("ggml-base.bin"), b"x").unwrap();

        let sections = call_default(dir);
        let models_section = sections
            .iter()
            .find(|s| s.id() == "models")
            .expect("debe existir la sección models");

        let mut items: Vec<MenuItemSpec> = Vec::new();
        for sec in models_section.build() {
            match sec {
                Section::Item(spec) => items.push(spec),
                Section::Submenu {
                    items: sub_items, ..
                } => items.extend(sub_items),
                Section::Separator => {}
            }
        }
        let tiny = items
            .iter()
            .find(|i| i.id == "model:ggml-tiny.bin")
            .expect("debe existir item ggml-tiny.bin");
        assert!(
            tiny.label.starts_with("✓ "),
            "tiny debe estar marcado como instalado: {}",
            tiny.label
        );

        let small = items
            .iter()
            .find(|i| i.id == "model:ggml-small.bin")
            .expect("debe existir item ggml-small.bin");
        assert!(
            small.label.starts_with("↓ Descargar ") || small.label.starts_with("↓ Download "),
            "small debe estar marcado como no instalado: {}",
            small.label
        );

        let base = items
            .iter()
            .find(|i| i.id == "model:ggml-base.bin")
            .expect("debe existir item ggml-base.bin");
        assert!(base.label.starts_with("✓ "));
        assert!(
            base.label.contains("← activo") || base.label.contains("← active"),
            "base debe estar marcado como activo: {}",
            base.label
        );
    }

    /// El ítem activo en UiLanguage lleva prefijo ✓ en la entry
    /// correspondiente.
    #[test]
    fn default_sections_marks_active_ui_language() {
        let (_tmp, dir) = empty_models_dir();
        let ctx = BuildContext {
            models_dir: dir,
            active_model: "ggml-base.bin".into(),
            ui_language: UiLanguage::Bilingual,
            theme: Theme::System,
            stt_mode: SttMode::Batch,
            prompt_preset: PromptPreset::BilingualEsEn,
            prompt_custom_text: String::new(),
            effort: EffortPreset::Balanced,
            model_lang_mismatch: None,
            input_devices: Vec::new(),
            input_device: None,
        };
        let sections = default_sections(&ctx);
        let sec = sections
            .iter()
            .find(|s| s.id() == "ui_language")
            .expect("debe existir ui_language");
        let items: Vec<MenuItemSpec> = sec
            .build()
            .into_iter()
            .flat_map(|s| match s {
                Section::Item(i) => vec![i],
                Section::Submenu { items, .. } => items,
                _ => vec![],
            })
            .collect();
        let bil = items.iter().find(|i| i.id == "ui_bil").unwrap();
        assert!(
            bil.label.starts_with("✓ "),
            "bilingüe debe estar activo: {}",
            bil.label
        );
        let es = items.iter().find(|i| i.id == "ui_es").unwrap();
        assert!(
            !es.label.starts_with("✓ "),
            "es no debe estar activo: {}",
            es.label
        );
    }

    /// El item prompt_custom muestra un preview del texto cuando NO
    /// está vacío.
    #[test]
    fn prompt_section_shows_custom_preview_when_set() {
        let (_tmp, dir) = empty_models_dir();
        let ctx = BuildContext {
            models_dir: dir,
            active_model: "ggml-base.bin".into(),
            ui_language: UiLanguage::Es,
            theme: Theme::System,
            stt_mode: SttMode::Batch,
            prompt_preset: PromptPreset::Custom,
            prompt_custom_text: "Dictaré kubernetes, gRPC y WASM".into(),
            effort: EffortPreset::Balanced,
            model_lang_mismatch: None,
            input_devices: Vec::new(),
            input_device: None,
        };
        let sections = default_sections(&ctx);
        let sec = sections.iter().find(|s| s.id() == "prompt").unwrap();
        let items: Vec<MenuItemSpec> = sec
            .build()
            .into_iter()
            .flat_map(|s| match s {
                Section::Item(i) => vec![i],
                Section::Submenu { items, .. } => items,
                _ => vec![],
            })
            .collect();
        let custom = items.iter().find(|i| i.id == "prompt_custom").unwrap();
        assert!(
            custom.label.contains("kubernetes"),
            "preview debe incluir el texto: {}",
            custom.label
        );
        assert!(
            custom.label.starts_with("✓ "),
            "custom activo: {}",
            custom.label
        );
    }

    /// Con ui_language=En, los labels de los submenús cambian.
    #[test]
    fn default_sections_with_ui_language_en_uses_english_labels() {
        let (_tmp, dir) = empty_models_dir();
        let ctx = BuildContext {
            models_dir: dir,
            active_model: "ggml-base.bin".into(),
            ui_language: UiLanguage::En,
            theme: Theme::System,
            stt_mode: SttMode::Batch,
            prompt_preset: PromptPreset::BilingualEsEn,
            prompt_custom_text: String::new(),
            effort: EffortPreset::Balanced,
            model_lang_mismatch: None,
            input_devices: Vec::new(),
            input_device: None,
        };
        let sections = default_sections(&ctx);
        let hotkey = sections.iter().find(|s| s.id() == "hotkey").unwrap();
        let items: Vec<MenuItemSpec> = hotkey
            .build()
            .into_iter()
            .filter_map(|s| match s {
                Section::Item(i) => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(items[0].label, "Change hotkey…");

        let theme = sections.iter().find(|s| s.id() == "theme").unwrap();
        let items: Vec<MenuItemSpec> = theme
            .build()
            .into_iter()
            .flat_map(|s| match s {
                Section::Submenu { items, .. } => items,
                _ => vec![],
            })
            .collect();
        assert!(items.iter().any(|i| i.label.contains("Dark")));
        assert!(items.iter().any(|i| i.label.contains("Light")));
        assert!(items.iter().any(|i| i.label.contains("System")));
    }

    /// Modelos `.en` llevan el sufijo de aviso.
    #[test]
    fn models_section_marks_en_only_models() {
        let (_tmp, dir) = empty_models_dir();
        std::fs::write(dir.join("ggml-tiny.en.bin"), b"x").unwrap();
        let sections = call_default(dir);
        let sec = sections.iter().find(|s| s.id() == "models").unwrap();
        let items: Vec<MenuItemSpec> = sec
            .build()
            .into_iter()
            .flat_map(|s| match s {
                Section::Item(i) => vec![i],
                Section::Submenu { items, .. } => items,
                _ => vec![],
            })
            .collect();
        let en = items
            .iter()
            .find(|i| i.id == "model:ggml-tiny.en.bin")
            .unwrap();
        assert!(
            en.label.contains("⚠") && en.label.contains("ingl") || en.label.contains("English"),
            "modelo .en debe llevar aviso: {}",
            en.label
        );
    }

    /// Con `model_lang_mismatch = None`, la sección de aviso no se añade
    /// al árbol (consistente con el caso normal de los tests previos).
    #[test]
    fn mismatch_section_absent_when_none() {
        let (_tmp, dir) = empty_models_dir();
        let sections = call_default(dir);
        assert!(
            sections.iter().all(|s| s.id() != "model_lang_mismatch"),
            "sin mismatch, no debe aparecer la sección de aviso"
        );
    }

    /// Con `model_lang_mismatch = Some(filename)`, la sección aparece
    /// **primero** en el árbol y renderiza un único item con id
    /// canónico `fix_model_lang` cuyo label incluye el filename
    /// sugerido.
    #[test]
    fn mismatch_section_present_and_first_when_active() {
        let (_tmp, dir) = empty_models_dir();
        let mut ctx = BuildContext {
            models_dir: dir,
            active_model: "ggml-small.en.bin".into(),
            ui_language: UiLanguage::Es,
            theme: Theme::System,
            stt_mode: SttMode::Batch,
            prompt_preset: PromptPreset::BilingualEsEn,
            prompt_custom_text: String::new(),
            effort: EffortPreset::Balanced,
            model_lang_mismatch: None,
            input_devices: Vec::new(),
            input_device: None,
        };
        ctx.model_lang_mismatch = Some("ggml-small.bin".into());

        let sections = default_sections(&ctx);
        assert_eq!(
            sections.len(),
            10,
            "8 base + 1 mismatch + 1 microphone = 10"
        );
        assert_eq!(sections[0].id(), "model_lang_mismatch");

        let items: Vec<MenuItemSpec> = sections[0]
            .build()
            .into_iter()
            .filter_map(|s| match s {
                Section::Item(i) => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(items.len(), 1, "exactamente 1 item de aviso");
        assert_eq!(items[0].id, "fix_model_lang");
        assert!(
            items[0].label.contains("ggml-small.bin"),
            "label debe incluir el filename sugerido: {}",
            items[0].label
        );
    }

    /// `id_to_action` mapea `"fix_model_lang"` a `MenuAction::FixModelLanguage`.
    #[test]
    fn fix_model_lang_action_is_mapped() {
        match id_to_action("fix_model_lang") {
            Some(MenuAction::FixModelLanguage(_)) => {}
            other => panic!("esperaba FixModelLanguage, obtuve {other:?}"),
        }
    }

    #[test]
    fn default_sections_cover_all_menu_actions() {
        let (_tmp, dir) = empty_models_dir();
        let sections = call_default(dir);
        let mut all_ids: Vec<String> = Vec::new();
        for s in &sections {
            for item in s.build() {
                match item {
                    Section::Item(spec) => all_ids.push(spec.id),
                    Section::Submenu { items, .. } => {
                        for spec in items {
                            if !spec.id.is_empty() {
                                all_ids.push(spec.id);
                            }
                        }
                    }
                    Section::Separator => {}
                }
            }
        }
        for expected in [
            "change_hotkey",
            "theme_dark",
            "theme_light",
            "theme_system",
            "mode_batch",
            "mode_streaming",
            "mode_chunked",
            "effort_balanced",
            "effort_robust",
            "effort_high_quality",
            "ui_es",
            "ui_en",
            "ui_bil",
            "prompt_bilingual",
            "prompt_es",
            "prompt_en",
            "prompt_custom",
            "prompt_edit",
            "open_models_dir",
            "check_updates",
            "exit",
            "mic_auto",
            "mic_reprobe",
        ] {
            assert!(
                all_ids.iter().any(|id| id == expected),
                "default_sections debe incluir item id {expected}"
            );
            assert!(
                id_to_action(expected).is_some(),
                "id_to_action debe mapear {expected}"
            );
        }
    }

    // -------------------------------------------------------------------
    // Tests específicos de MicrophoneSection
    // -------------------------------------------------------------------

    /// Items esperados del submenú Micrófono:
    ///   1 (mic_auto) + N dispositivos + 1 separador + 1 (mic_reprobe) = N + 3.
    fn mic_items(section: &MicrophoneSection) -> Vec<MenuItemSpec> {
        let built = section.build();
        assert_eq!(built.len(), 1, "debe haber exactamente 1 submenú");
        match built.into_iter().next().unwrap() {
            Section::Submenu { items, .. } => items,
            _ => panic!("se esperaba un submenú"),
        }
    }

    #[test]
    fn microphone_section_marks_auto_active_when_none() {
        let sec = MicrophoneSection::new(
            vec![MicDevice {
                name: "USB Mic".into(),
                is_default: true,
            }],
            None,
            UiLanguage::Es,
        );
        let items = mic_items(&sec);
        assert_eq!(items[0].id, "mic_auto");
        assert!(items[0].label.starts_with("✓ "));
        // Dispositivo NO debe tener marca (auto está activo).
        assert!(items.iter().any(|i| i.id == "mic:USB Mic"));
        let usb = items.iter().find(|i| i.id == "mic:USB Mic").unwrap();
        assert!(!usb.label.starts_with("✓ "));
    }

    #[test]
    fn microphone_section_marks_selected_device_active() {
        let sec = MicrophoneSection::new(
            vec![
                MicDevice {
                    name: "USB Mic".into(),
                    is_default: true,
                },
                MicDevice {
                    name: "Headset".into(),
                    is_default: false,
                },
            ],
            Some("Headset".into()),
            UiLanguage::Es,
        );
        let items = mic_items(&sec);
        // Auto NO activo.
        assert!(!items[0].label.starts_with("✓ "));
        // "Headset" activo.
        let headset = items.iter().find(|i| i.id == "mic:Headset").unwrap();
        assert!(headset.label.starts_with("✓ "));
        assert!(headset.label.contains("← activo"));
        // "USB Mic" (default) NO activo y debe llevar la marca "(default)".
        let usb = items.iter().find(|i| i.id == "mic:USB Mic").unwrap();
        assert!(!usb.label.starts_with("✓ "));
        assert!(usb.label.contains("(default)"));
    }

    #[test]
    fn microphone_section_shows_empty_message_when_no_devices() {
        let sec = MicrophoneSection::new(Vec::new(), None, UiLanguage::Es);
        let items = mic_items(&sec);
        // 1 auto + 1 empty (deshabilitado) + 1 separator + 1 reprobe = 4
        assert_eq!(items.len(), 4);
        assert!(items[1].label.contains("No se detectaron"));
        assert!(!items[1].enabled);
        assert_eq!(items[3].id, "mic_reprobe");
    }

    #[test]
    fn microphone_section_in_default_sections() {
        let (_tmp, dir) = empty_models_dir();
        let mut ctx = BuildContext::initial(dir, "ggml-base.bin");
        ctx.input_devices = vec![MicDevice {
            name: "USB Mic".into(),
            is_default: true,
        }];
        ctx.input_device = Some("USB Mic".into());
        let sections = default_sections(&ctx);
        let mic = sections
            .iter()
            .find(|s| s.id() == "microphone")
            .expect("MicrophoneSection debe estar en default_sections");
        let items = mic_items_as_dyn(mic.as_ref());
        assert!(items.iter().any(|i| i.id == "mic:USB Mic"));
        assert!(items.iter().any(|i| i.id == "mic_auto"));
        assert!(items.iter().any(|i| i.id == "mic_reprobe"));
    }

    /// Helper para extraer items de un `&dyn MenuSection` en los tests
    /// que operan sobre la composición (no sobre el struct concreto).
    fn mic_items_as_dyn(section: &dyn MenuSection) -> Vec<MenuItemSpec> {
        let built = section.build();
        assert_eq!(built.len(), 1);
        match built.into_iter().next().unwrap() {
            Section::Submenu { items, .. } => items,
            _ => panic!("se esperaba un submenú"),
        }
    }
}
