"use strict";

// Orbit class of each satellite, keyed by NORAD catalogue number.
const ORBIT_CLASS = { 64541: "LEO", 59598: "MEO", 38652: "GEO", 42738: "IGSO" };

const HINTS = {
  fixed:
    "The camera turns with the Earth. Ground tracks show where each satellite " +
    "passes overhead.",
  inertial:
    "The camera stays fixed relative to the stars and the Earth turns underneath. " +
    "GEO and IGSO orbits are the same size, only tilted differently.",
};

const MODEL_HINTS = {
  kepler: "Two-body Kepler propagation: solid orbits, filled dots.",
  sgp4:
    "SGP4/SDP4 from the same GP elements: dashed orbits, rings. It adds Earth's " +
    "oblateness, drag and lunar and solar gravity.",
  both:
    "Filled dot and solid orbit: Kepler. Ring and dashed orbit: SGP4. Δ is the " +
    "distance between them. Follow PULSAR-0 to see LEO drift apart.",
};

const WHOLE_EARTH_HEIGHT_M = 9e7; // far enough to see the GEO ring
const LIVE_UPDATE_MS = 250;

const viewer = new Cesium.Viewer("globe", {
  // Imagery bundled with CesiumJS: no account or access token needed.
  baseLayer: Cesium.ImageryLayer.fromProviderAsync(
    Cesium.TileMapServiceImageryProvider.fromUrl(
      Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"),
    ),
  ),
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  shouldAnimate: true,
});

const ui = {
  list: document.getElementById("sat-list"),
  fixed: document.getElementById("frame-fixed"),
  inertial: document.getElementById("frame-inertial"),
  hint: document.getElementById("frame-hint"),
  models: {
    kepler: document.getElementById("model-kepler"),
    sgp4: document.getElementById("model-sgp4"),
    both: document.getElementById("model-both"),
  },
  modelHint: document.getElementById("model-hint"),
  tracks: document.getElementById("show-tracks"),
  orbits: document.getElementById("show-orbits"),
  freeCamera: document.getElementById("free-camera"),
  loading: document.getElementById("loading"),
};

let frame = "fixed";
let model = "both";
let inertialReset = false; // next inertial camera update jumps to the whole-Earth view
const satellites = []; // { entity, track, orbit, sgp4, sgp4orbit, button, live }

// ---------- Camera in the inertial frame ----------

function icrfToFixed(time) {
  // Without preloaded Earth-orientation data Cesium returns undefined here and
  // we fall back to a GMST-only rotation, the same model as frames.py.
  return (
    Cesium.Transforms.computeIcrfToFixedMatrix(time) ??
    Cesium.Transforms.computeTemeToPseudoFixedMatrix(time)
  );
}

function followInertialFrame(_scene, time) {
  const camera = viewer.camera;
  const transform = Cesium.Matrix4.fromRotationTranslation(icrfToFixed(time));
  let offset = Cesium.Cartesian3.clone(camera.position);
  if (inertialReset) {
    offset = Cesium.Cartesian3.multiplyByScalar(
      Cesium.Cartesian3.normalize(offset, offset),
      WHOLE_EARTH_HEIGHT_M + 6.4e6,
      offset,
    );
    inertialReset = false;
  }
  camera.lookAtTransform(transform, offset);
}

function setFrame(next) {
  if (next === frame) return;
  frame = next;
  ui.fixed.setAttribute("aria-pressed", String(frame === "fixed"));
  ui.inertial.setAttribute("aria-pressed", String(frame === "inertial"));
  ui.hint.textContent = HINTS[frame];
  if (frame === "inertial") {
    viewer.trackedEntity = undefined; // tracking and the inertial camera would fight
    viewer.scene.postUpdate.addEventListener(followInertialFrame);
  } else {
    viewer.scene.postUpdate.removeEventListener(followInertialFrame);
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  }
}

function showWholeEarth() {
  viewer.trackedEntity = undefined;
  if (frame === "inertial") {
    inertialReset = true;
    return;
  }
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(60, 15, WHOLE_EARTH_HEIGHT_M),
    duration: 1.5,
  });
}

// ---------- Satellite list ----------

function shortName(name) {
  return name.split(" (")[0];
}

function viewFromBehind(position) {
  // Camera on the far side of the satellite, so the Earth is behind it.
  // Cesium's tracking frame depends on the orbit (EntityView.js): beyond
  // 1.25 Earth radii x points to the Earth (nadir) and z north; closer in
  // (LEO) z points away from the Earth and x along the track.
  const r = Cesium.Cartesian3.magnitude(position);
  const h = r - Cesium.Ellipsoid.WGS84.maximumRadius;
  if (r > 1.25 * Cesium.Ellipsoid.WGS84.maximumRadius) {
    return new Cesium.Cartesian3(-0.6 * h, 0, 0.15 * h);
  }
  return new Cesium.Cartesian3(-3 * h, 0, 5 * h); // LEO: high enough to see the curve
}

function follow(sat) {
  // Cesium cannot track a hidden entity, so follow the SGP4 ring in SGP4 mode.
  const target = model === "sgp4" ? sat.sgp4 : sat.entity;
  setFrame("fixed");
  viewer.selectedEntity = target;
  viewer.trackedEntity = target;
}

// ---------- Layers ----------

function applyVisibility() {
  const kepler = model !== "sgp4";
  const sgp4 = model !== "kepler";
  for (const s of satellites) {
    s.entity.show = kepler;
    s.orbit.show = kepler && ui.orbits.checked;
    s.track.show = kepler && ui.tracks.checked; // ground tracks are Kepler only
    s.sgp4.show = sgp4;
    s.sgp4orbit.show = sgp4 && ui.orbits.checked;
  }
  ui.tracks.disabled = !kepler;
}

function setModel(next) {
  model = next;
  for (const [name, button] of Object.entries(ui.models)) {
    button.setAttribute("aria-pressed", String(name === model));
  }
  ui.modelHint.textContent = MODEL_HINTS[model];
  // Keep following the same satellite if its tracked marker was just hidden.
  const tracked = satellites.find(
    (s) => viewer.trackedEntity === s.entity || viewer.trackedEntity === s.sgp4,
  );
  applyVisibility();
  if (tracked && !viewer.trackedEntity.show) follow(tracked);
}

function addSatellite(ds, entity, start) {
  const norad = entity.id.slice("sat-".length);
  const color = entity.point.color.getValue(start).toCssColorString();

  entity.viewFrom = viewFromBehind(entity.position.getValue(start));
  const sgp4 = ds.entities.getById(`sgp4-${norad}`);
  sgp4.viewFrom = viewFromBehind(sgp4.position.getValue(start));

  const button = document.createElement("button");
  button.type = "button";
  button.className = "sat";
  button.setAttribute("aria-pressed", "false");
  button.innerHTML = `
    <span class="swatch" style="background:${color}"></span>
    <span class="sat-name"></span>
    <span class="badge">${ORBIT_CLASS[norad] ?? ""}</span>
    <span class="sat-live">&nbsp;</span>`;
  button.querySelector(".sat-name").textContent = shortName(entity.name);

  const li = document.createElement("li");
  li.append(button);
  ui.list.append(li);

  const sat = {
    entity,
    track: ds.entities.getById(`track-${norad}`),
    orbit: ds.entities.getById(`orbit-${norad}`),
    sgp4,
    sgp4orbit: ds.entities.getById(`sgp4orbit-${norad}`),
    button,
    live: button.querySelector(".sat-live"),
  };
  button.addEventListener("click", () => follow(sat));
  satellites.push(sat);
}

// ---------- Live altitude and speed ----------

const scratch = {
  before: new Cesium.JulianDate(),
  after: new Cesium.JulianDate(),
  a: new Cesium.Cartesian3(), // used by inertialSpeedKmS
  b: new Cesium.Cartesian3(),
  kepler: new Cesium.Cartesian3(), // used by updateLive
  sgp4: new Cesium.Cartesian3(),
};

function inertialSpeedKmS(sat, time) {
  // Difference over 2 s in the inertial frame: the orbital speed, the same
  // quantity as in the report (not the speed relative to the ground). The
  // window is shifted at the ends of the day so both points have data.
  const { before, after, a, b } = scratch;
  const { startTime, stopTime } = viewer.clock;
  const fromStart = Cesium.JulianDate.secondsDifference(time, startTime);
  const toStop = Cesium.JulianDate.secondsDifference(stopTime, time);
  const shift = fromStart < 1 ? 1 - fromStart : toStop < 1 ? toStop - 1 : 0;
  Cesium.JulianDate.addSeconds(time, shift - 1, before);
  Cesium.JulianDate.addSeconds(time, shift + 1, after);
  const frameI = Cesium.ReferenceFrame.INERTIAL;
  const p0 = sat.orbit.position.getValueInReferenceFrame(before, frameI, a);
  const p1 = sat.orbit.position.getValueInReferenceFrame(after, frameI, b);
  if (!p0 || !p1) return undefined;
  return Cesium.Cartesian3.distance(p0, p1) / 2 / 1000;
}

let lastLiveUpdate = 0;
function updateLive(clock) {
  const now = performance.now();
  if (now - lastLiveUpdate < LIVE_UPDATE_MS) return;
  lastLiveUpdate = now;
  for (const sat of satellites) {
    const pos = sat.entity.position.getValue(clock.currentTime, scratch.kepler);
    const pos4 = sat.sgp4.position.getValue(clock.currentTime, scratch.sgp4);
    const v = inertialSpeedKmS(sat, clock.currentTime);
    if (!pos || !pos4 || v === undefined) continue;
    // Altitude and speed of the Kepler satellite; Δ is the Kepler-SGP4 gap.
    const altKm = Cesium.Cartographic.fromCartesian(pos).height / 1000;
    const gapKm = Cesium.Cartesian3.distance(pos, pos4) / 1000;
    sat.live.textContent =
      `${Math.round(altKm).toLocaleString("en-US")} km · ${v.toFixed(3)} km/s` +
      ` · Δ ${Math.round(gapKm)} km`;
  }
}

// ---------- Wiring ----------

ui.fixed.addEventListener("click", () => setFrame("fixed"));
ui.inertial.addEventListener("click", () => setFrame("inertial"));
ui.freeCamera.addEventListener("click", showWholeEarth);
ui.tracks.addEventListener("change", applyVisibility);
ui.orbits.addEventListener("change", applyVisibility);
for (const [name, button] of Object.entries(ui.models)) {
  button.addEventListener("click", () => setModel(name));
}

viewer.trackedEntityChanged.addEventListener((entity) => {
  if (entity && frame === "inertial") setFrame("fixed");
  for (const s of satellites) {
    const pressed = entity !== undefined && (s.entity === entity || s.sgp4 === entity);
    s.button.setAttribute("aria-pressed", String(pressed));
  }
});

async function main() {
  viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(60, 15, WHOLE_EARTH_HEIGHT_M),
  });
  try {
    const ds = await Cesium.CzmlDataSource.load("orbits.czml");
    await viewer.dataSources.add(ds); // the viewer adopts the CZML clock
    const start = ds.clock.startTime;
    const day = Cesium.JulianDate.toDate(start);
    document.getElementById("window-date").textContent = `, ${day.toLocaleDateString(
      "en-GB",
      { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" },
    )}`;
    for (const entity of ds.entities.values) {
      if (entity.id.startsWith("sat-")) addSatellite(ds, entity, start);
    }
    setModel(model);
    viewer.clock.onTick.addEventListener(updateLive);
    updateLive(viewer.clock);
    ui.loading.hidden = true;
  } catch (err) {
    console.error(err);
    ui.loading.textContent =
      "Could not load orbits.czml. Serve this folder over HTTP (see README).";
  }
}

main();
