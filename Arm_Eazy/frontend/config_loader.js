// =============================
// CONFIG LOADER MODULE
// =============================
// Applies a loaded arm config to the scene. RobotState owns joint axes,
// types, limits, sliders, badges, and per-joint rotation/translation
// rendering (see robot_state.js). This file handles the parts RobotState
// doesn't: X3D model file paths, and the base translation offset for every
// part in the chain (including Joint1..6 — for "rotate" joints this is the
// ONLY place that offset gets set, since RobotState.render() only touches
// the `rotation` attribute for those; for "translate" joints RobotState
// recomputes translation from this same base value plus the joint's
// current value, so setting it here first is harmless and keeps behavior
// correct even before the first render() call).

function setInlineUrl(transformId, path) {
  const tf = document.getElementById(transformId);
  if (!tf) return;

  const inline = tf.querySelector("Inline");
  if (!inline) return;

  inline.setAttribute("url", `"${path}"`);
}

function setTransformTranslation(transformId, vec3) {
  const tf = document.getElementById(transformId);
  if (!tf) return;

  const [x, y, z] = window.SharedUtils ? window.SharedUtils.parseVec3(vec3) : vec3;
  tf.setAttribute("translation", `${x} ${y} ${z}`);
}

function applyArmConfig(cfg) {

  const parts = [
    "BaseTransform","Joint1","Joint2","Joint3",
    "Joint4","Joint5","Joint6","Ik_bone"
  ];

  // ===== MODEL PATHS =====
  if (cfg.model_paths) {
    parts.forEach(part => {
      if (cfg.model_paths[part]) {
        setInlineUrl(part, cfg.model_paths[part]);
      }
    });
  }

  // ===== BASE TRANSLATIONS (see file header for why this covers all parts) =====
  if (cfg.translations_mm) {
    parts.forEach(part => {
      if (cfg.translations_mm[part]) {
        setTransformTranslation(part, cfg.translations_mm[part]);
      }
    });
  }

  // ===== JOINTS: axes, type, limits, sliders, badges, rotation/translation =====
  if (window.RobotState) {
    window.RobotState.setConfig(cfg);
  } else {
    console.warn("[config_loader] RobotState not loaded — joints will not update");
  }

  // ===== PUSH TO BACKEND (and let robot_bridge.js know the new config) =====
  if (window.setRobotConfig) {
    window.setRobotConfig(cfg);
  }

  const status = document.getElementById("configStatus");
  if (status) status.textContent = "Config applied ✔";
}


// =============================
// FILE INPUT HANDLER
// =============================

function setupConfigLoader() {
  const input = document.getElementById("configFileInput");
  const status = document.getElementById("configStatus");

  if (!input) return;

  input.addEventListener("change", async () => {
    try {
      if (!input.files || !input.files[0]) return;

      const text = await input.files[0].text();
      const cfg = JSON.parse(text);

      if (!cfg || typeof cfg !== "object") {
        throw new Error("Invalid JSON structure.");
      }

      if (status) status.textContent = "Loading...";
      applyArmConfig(cfg);

    } catch (err) {
      if (status) status.textContent = "Error: " + err.message;
    }
  });
}

// =============================
// DEFAULT CONFIG BOOTSTRAP
// =============================
// Load config/default_config.json automatically on startup, same as
// config_editor.html does, so the twin doesn't rely solely on the user
// picking a file before anything works.

async function loadDefaultConfigOnStartup() {
  try {
    const res = await fetch("../../config/default_config.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const cfg = await res.json();
    applyArmConfig(cfg);
  } catch (err) {
    console.warn("[config_loader] Could not load default_config.json, using SharedUtils fallback:", err.message);
    // RobotState/kinematics.js already fall back to window.SharedUtils.FALLBACK_GEOMETRY
    // on their own when state.config is null, so no further action is needed here.
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupConfigLoader();
  loadDefaultConfigOnStartup();
});
