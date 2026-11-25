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

const MODE_HINTS = {
  day:
    "One UTC day, computed in advance with Kepler and SGP4, with the day's passes.",
  now:
    "Real time, from now. Drag the timeline to look ahead or back; the next " +
    "passes follow the clock.",
};

const WHOLE_EARTH_HEIGHT_M = 9e7; // far enough to see the GEO ring
const LIVE_UPDATE_MS = 250;
const SHORT_PASS_S = 15 * 60; // LEO passes: slow the clock down to watch them
const SHORT_PASS_MULTIPLIER = 30;

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
  stationSection: document.getElementById("station-section"),
  stationTitle: document.getElementById("station-title"),
  stationHint: document.getElementById("station-hint"),
  stationPicker: document.getElementById("station-picker"),
  noPasses: document.getElementById("no-passes"),
  passList: document.getElementById("pass-list"),
  showStation: document.getElementById("show-station"),
  modeDay: document.getElementById("mode-day"),
  modeNow: document.getElementById("mode-now"),
  modeHint: document.getElementById("mode-hint"),
  subtitleDay: document.getElementById("subtitle-day"),
  subtitleNow: document.getElementById("subtitle-now"),
};

let frame = "fixed";
let model = "both";
let mode = "day"; // "day": the CZML day; "now": live SGP4 in the browser
let inertialReset = false; // next inertial camera update jumps to the whole-Earth view
// { entity, track, orbit, sgp4, sgp4orbit, button, live, sky } from the CZML,
// plus { gp, satrec, epochJd, liveEntity, liveOrbit } for the live mode
const satellites = [];
// Ground stations: [{ slug, entity, name, maskDeg, lon, lat, toLocal, rings,
// links, passes: [{ ..., riseJd, setJd }] }]; `station` is the one shown.
const stations = [];
let station;
let passRows = []; // [{ pass, row }] for the shown station
let defaultMultiplier; // the CZML clock speed, restored after a slowed-down pass
let slowPass; // the pass currently playing at SHORT_PASS_MULTIPLIER

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

function shortName(entity, time) {
  // The globe label carries the short name from style.py (e.g. "GALILEO 29"),
  // the same one the pass list uses.
  return entity.label?.text?.getValue(time) ?? entity.name.split(" (")[0];
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
  // Cesium cannot track a hidden entity, so follow the SGP4 ring in SGP4 mode,
  // and the live marker in the live mode.
  const target =
    mode === "now" ? sat.liveEntity : model === "sgp4" ? sat.sgp4 : sat.entity;
  setFrame("fixed");
  viewer.selectedEntity = target;
  viewer.trackedEntity = target;
}

// ---------- Layers ----------

function applyVisibility() {
  const live = mode === "now";
  const kepler = !live && model !== "sgp4";
  const sgp4 = !live && model !== "kepler";
  for (const s of satellites) {
    s.entity.show = kepler;
    s.orbit.show = kepler && ui.orbits.checked;
    s.track.show = kepler && ui.tracks.checked; // ground tracks are Kepler only
    s.sgp4.show = sgp4;
    s.sgp4orbit.show = sgp4 && ui.orbits.checked;
    if (s.liveEntity) {
      s.liveEntity.show = live;
      s.liveOrbit.show = live && ui.orbits.checked;
    }
  }
  ui.tracks.disabled = !kepler;
  for (const button of Object.values(ui.models)) button.disabled = live;
  ui.modelHint.textContent = live
    ? "The live mode uses SGP4 only."
    : MODEL_HINTS[model];
  applyStationVisibility();
}

function setModel(next) {
  model = next;
  for (const [name, button] of Object.entries(ui.models)) {
    button.setAttribute("aria-pressed", String(name === model));
  }
  // Keep following the same satellite if its tracked marker was just hidden.
  const tracked = trackedSatellite();
  applyVisibility();
  if (tracked && !viewer.trackedEntity.show) follow(tracked);
}

function trackedSatellite() {
  const e = viewer.trackedEntity;
  return e && satellites.find((s) => [s.entity, s.sgp4, s.liveEntity].includes(e));
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
    <span class="sat-live">&nbsp;</span>
    <span class="sat-sky" hidden></span>`;
  button.querySelector(".sat-name").textContent = shortName(entity, start);

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
    sky: button.querySelector(".sat-sky"),
    norad: Number(norad),
    color,
  };
  button.addEventListener("click", () => follow(sat));
  satellites.push(sat);
}

// ---------- Ground station ----------

function utcClock(iso) {
  return iso.slice(11, 16);
}

function passLabel(p) {
  if (!p.risesInWindow && !p.setsInWindow) return "All day";
  const rise = p.risesInWindow ? utcClock(p.rise) : "00:00";
  const set = p.setsInWindow ? utcClock(p.set) : "24:00";
  return `${rise}–${set}`;
}

function jumpToPass(p) {
  const sat = satellites.find((s) => s.norad === p.norad);
  const rise = Cesium.JulianDate.fromIso8601(p.rise);
  const set = Cesium.JulianDate.fromIso8601(p.set);
  const duration = Cesium.JulianDate.secondsDifference(set, rise);
  const lead = Math.min(60, duration / 10);
  const start = Cesium.JulianDate.addSeconds(rise, -lead, new Cesium.JulianDate());
  viewer.clock.currentTime = Cesium.JulianDate.greaterThan(start, viewer.clock.startTime)
    ? start
    : viewer.clock.startTime.clone();
  slowPass = duration < SHORT_PASS_S ? p : undefined;
  viewer.clock.multiplier = slowPass ? SHORT_PASS_MULTIPLIER : defaultMultiplier;
  viewer.clock.shouldAnimate = true;
  // Look down on the station from high enough to see the satellite's side of
  // the sky: a few thousand km for LEO, much further for higher orbits.
  setFrame("fixed");
  viewer.trackedEntity = undefined;
  viewer.selectedEntity = model === "kepler" ? sat?.entity : sat?.sgp4;
  const leo = sat && Cesium.Cartographic.fromCartesian(
    sat.entity.position.getValue(rise),
  ).height < 2e6;
  // South of the station looking north, so the map stays north-up; clamped so
  // a station far south never puts the camera past the pole.
  const { lon, lat } = station;
  const cameraLat = Math.max(lat - (leo ? 12 : 35), -89);
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, cameraLat, leo ? 4.5e6 : 3.5e7),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(leo ? -60 : -75), roll: 0 },
    duration: 1.5,
  });
}

function addPassRow(pass) {
  const sat = satellites.find((s) => s.norad === pass.norad);
  const row = document.createElement("button");
  row.type = "button";
  row.className = "pass";
  row.innerHTML = `
    <span class="swatch" style="background:${sat?.color ?? "#999"}"></span>
    <span class="pass-name"></span>
    <span class="pass-time">${passLabel(pass)}</span>
    <span class="pass-max">max ${Math.round(pass.maxEl)}°</span>`;
  row.querySelector(".pass-name").textContent = pass.name;
  row.addEventListener("click", () => jumpToPass(pass));
  const li = document.createElement("li");
  li.append(row);
  ui.passList.append(li);
  return { pass, row };
}

// "ring-new-york-64541" belongs to station "new-york", not "new": compare the
// whole slug between the kind and the NORAD number, not just a prefix.
function ownedBy(id, kind, slug) {
  const m = id.match(/^(ring|link)-(.+)-(\d+)$/);
  return m !== null && m[1] === kind && m[2] === slug;
}

function readStation(ds, entity, start) {
  const props = entity.properties.getValue(start);
  const position = entity.position.getValue(start);
  const carto = Cesium.Cartographic.fromCartesian(position);
  const slug = entity.id.slice("station-".length);
  return {
    slug,
    entity,
    order: props.order,
    name: entity.name.replace(/ ground station$/, ""),
    maskDeg: props.maskDeg,
    lon: Cesium.Math.toDegrees(carto.longitude),
    lat: Cesium.Math.toDegrees(carto.latitude),
    // Earth-fixed position -> station East-North-Up
    toLocal: Cesium.Matrix4.inverseTransformation(
      Cesium.Transforms.eastNorthUpToFixedFrame(position),
      new Cesium.Matrix4(),
    ),
    rings: ds.entities.values.filter((e) => ownedBy(e.id, "ring", slug)),
    links: ds.entities.values.filter((e) => ownedBy(e.id, "link", slug)),
    passes: JSON.parse(props.passes).map((p) => ({
      ...p,
      riseJd: Cesium.JulianDate.fromIso8601(p.rise),
      setJd: Cesium.JulianDate.fromIso8601(p.set),
    })),
  };
}

function setupStations(ds, start) {
  for (const entity of ds.entities.values) {
    if (entity.id.startsWith("station-")) stations.push(readStation(ds, entity, start));
  }
  if (!stations.length) return;
  stations.sort((a, b) => a.order - b.order);
  for (const st of stations) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = st.name;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => selectStation(st.slug));
    ui.stationPicker.append(button);
    st.button = button;
  }
  for (const s of satellites) s.sky.hidden = false;
  ui.stationSection.hidden = false;
  // A link like …/#tokyo opens on that station.
  selectStation(stationFromHash() ?? stations[0].slug);
  // Editing the hash in an open page (or following a #quito link) switches too.
  window.addEventListener("hashchange", () => {
    const slug = stationFromHash() ?? stations[0].slug;
    if (slug !== station.slug) selectStation(slug);
  });
}

function stationFromHash() {
  const slug = location.hash.slice(1).toLowerCase();
  return stations.some((s) => s.slug === slug) ? slug : undefined;
}

function selectStation(slug) {
  if (station?.slug === slug) return; // already shown: nothing to rebuild
  const previous = station;
  station = stations.find((s) => s.slug === slug);
  if (previous) {
    // Don't leave the old station's marker or ring in the info box, or its
    // slowed-down pass running at 30x.
    const old = [previous.entity, ...previous.rings, ...previous.links];
    if (old.includes(viewer.selectedEntity)) viewer.selectedEntity = undefined;
    if (slowPass) {
      if (viewer.clock.multiplier === SHORT_PASS_MULTIPLIER) {
        viewer.clock.multiplier = defaultMultiplier;
      }
      slowPass = undefined;
    }
  }
  for (const st of stations) st.button.setAttribute("aria-pressed", String(st === station));
  ui.passList.replaceChildren();
  passRows = station.passes.map(addPassRow);
  nextPasses = undefined;
  showStationText();
  if (location.hash.slice(1) !== slug) {
    const hash = slug === stations[0].slug ? "" : `#${slug}`;
    history.replaceState(null, "", location.pathname + location.search + hash);
  }
  applyStationVisibility(); // also covers a checkbox the browser restored
  lastLiveUpdate = 0; // refresh the sky lines now, even with the clock paused
  updateLive(viewer.clock);
}

function showStationText() {
  // The day's pass list in the day mode; in the live mode each satellite's
  // line says where it is now and when it next rises or sets.
  const live = mode === "now";
  ui.stationTitle.textContent = live
    ? `Over ${station.name} now`
    : `Passes over ${station.name}`;
  ui.stationHint.textContent = live
    ? `Above ${station.maskDeg}° elevation, from SGP4. Under each satellite: ` +
      "its elevation now and its next rise or set, within 24 hours."
    : `Above ${station.maskDeg}° elevation, from SGP4, times in UTC. Click a pass ` +
      "to jump to it. Short LEO passes play slower.";
  ui.passList.hidden = live;
  ui.noPasses.hidden = live || passRows.length > 0;
}

function applyStationVisibility() {
  const show = ui.showStation.checked;
  for (const st of stations) {
    const on = show && st === station;
    st.entity.show = on;
    for (const e of st.rings) e.show = on;
    // Links end at the SGP4 marker (passes come from SGP4): hide them with it.
    // They follow the day's passes, so not in the live mode either.
    for (const e of st.links) e.show = on && model !== "kepler" && mode === "day";
  }
}

function lookAngles(positionFixed) {
  const local = Cesium.Matrix4.multiplyByPoint(
    station.toLocal,
    positionFixed,
    scratch.local,
  );
  const range = Cesium.Cartesian3.magnitude(local);
  const el = Cesium.Math.toDegrees(Math.asin(local.z / range));
  const az = (Cesium.Math.toDegrees(Math.atan2(local.x, local.y)) + 360) % 360;
  return { az, el };
}

function updateStation(sat, pos4) {
  // Always SGP4, like the pass list, so a row and its pass never disagree.
  // ">=" as in visibility.find_passes.
  const { az, el } = lookAngles(pos4);
  const up = el >= station.maskDeg;
  sat.sky.classList.toggle("up", up);
  const deg = (x) => `${Math.round(x)}°`.replace("-", "−");
  sat.sky.textContent = up
    ? `${station.name}: ${deg(el)} up, azimuth ${deg(az)}`
    : `${station.name}: ${deg(el)}, below the mask`;
}

function updatePassRows(time) {
  // Back to normal speed once the clock leaves the slowed-down pass (it ended,
  // or the user scrubbed away), unless the user has changed the speed.
  const outside =
    slowPass &&
    (Cesium.JulianDate.greaterThan(time, slowPass.setJd) ||
      Cesium.JulianDate.secondsDifference(slowPass.riseJd, time) > 120);
  if (outside) {
    if (viewer.clock.multiplier === SHORT_PASS_MULTIPLIER) {
      viewer.clock.multiplier = defaultMultiplier;
    }
    slowPass = undefined;
  }
  for (const { pass, row } of passRows) {
    const active =
      !Cesium.JulianDate.lessThan(time, pass.riseJd) &&
      !Cesium.JulianDate.greaterThan(time, pass.setJd);
    row.setAttribute("aria-current", String(active));
  }
}

// ---------- Live altitude and speed ----------

const scratch = {
  before: new Cesium.JulianDate(),
  after: new Cesium.JulianDate(),
  a: new Cesium.Cartesian3(), // used by inertialSpeedKmS
  b: new Cesium.Cartesian3(),
  kepler: new Cesium.Cartesian3(), // used by updateLive
  sgp4: new Cesium.Cartesian3(),
  local: new Cesium.Cartesian3(), // used by lookAngles
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
  if (mode === "now") {
    updateNow(clock.currentTime);
    return;
  }
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
    if (station) updateStation(sat, pos4);
  }
  if (station) updatePassRows(clock.currentTime);
}

// ---------- Live mode: SGP4 in the browser ----------

// satellite.js is published as ES modules only; jsDelivr's +esm bundles it
// into one file, which this classic script can import(). If it (or gp.json)
// doesn't load, the live mode stays off and the day view works as before.
const SATELLITE_JS = "https://cdn.jsdelivr.net/npm/satellite.js@7.1.0/+esm";
const NEXT_PASS_HORIZON_S = 24 * 3600;
const NEXT_PASS_STEP_S = 30;
const NEXT_PASS_REFRESH_S = 60; // of clock time
const PEAK_STEP_S = 2; // the top of a pass shorter than PEAK_SEARCH_MAX_S
const PEAK_SEARCH_MAX_S = 2 * 3600;
const ORBIT_SAMPLES = 180;
const ORBIT_REFRESH_S = 600; // of clock time

let sgp4lib; // satellite.js, once loaded
let liveState = "loading"; // "loading", "ready" or "unavailable"
let dayClock; // the CZML clock, put back when leaving the live mode
let nextPasses; // { slug, at, bySat: Map(norad -> next pass) } for the station

const liveScratch = {
  teme: new Cesium.Matrix3(),
  cartesian: new Cesium.Cartesian3(),
  position: new Cesium.Cartesian3(),
  time: new Cesium.JulianDate(),
};

function sgp4At(sat, time) {
  // TEME position [km] and velocity [km/s], or undefined if SGP4 fails.
  // satellite.js's propagate() takes a JS Date, which keeps milliseconds (and
  // json2satrec reads the epoch into one too): up to 1 ms, about 7 m along a
  // LEO orbit. Minutes since the epoch from JulianDates (both from the text,
  // to the microsecond) go straight to sgp4(), as Python's sgp4 does.
  const minutes = Cesium.JulianDate.secondsDifference(time, sat.epochJd) / 60;
  const r = sgp4lib.sgp4(sat.satrec, minutes);
  return r && typeof r.position === "object" ? r : undefined;
}

function temeToFixed(km, matrix, result) {
  const c = Cesium.Cartesian3.fromElements(
    km.x * 1000,
    km.y * 1000,
    km.z * 1000,
    liveScratch.cartesian,
  );
  return Cesium.Matrix3.multiplyByVector(matrix, c, result ?? new Cesium.Cartesian3());
}

function liveFixedPosition(sat, time, result) {
  // Earth-fixed position [m]: TEME turned by GMST, as Cesium does elsewhere
  // without Earth-orientation data (and frames.py does in Python).
  const r = sgp4At(sat, time);
  if (!r) return undefined;
  const m = Cesium.Transforms.computeTemeToPseudoFixedMatrix(time, liveScratch.teme);
  return temeToFixed(r.position, m, result);
}

function liveOrbitPositions(sat, time) {
  // One revolution of TEME positions, drawn with the Earth's rotation at the
  // current time: the orbit in space, as in the day view. Recomputed every
  // ORBIT_REFRESH_S of clock time (it changes slowly), rotated every frame.
  let o = sat.liveOrbitCache;
  if (!o || Math.abs(Cesium.JulianDate.secondsDifference(time, o.at)) > ORBIT_REFRESH_S) {
    const periodS = ((2 * Math.PI) / sat.satrec.no) * 60; // no: rad/min
    const teme = [];
    for (let k = 0; k <= ORBIT_SAMPLES; k++) {
      const t = Cesium.JulianDate.addSeconds(
        time,
        (periodS * k) / ORBIT_SAMPLES,
        liveScratch.time,
      );
      const r = sgp4At(sat, t);
      if (r) teme.push(r.position);
    }
    o = sat.liveOrbitCache = {
      at: time.clone(),
      teme,
      fixed: teme.map(() => new Cesium.Cartesian3()),
    };
  }
  const m = Cesium.Transforms.computeTemeToPseudoFixedMatrix(time, liveScratch.teme);
  return o.teme.map((p, i) => temeToFixed(p, m, o.fixed[i]));
}

function addLiveEntities(sat) {
  const color = Cesium.Color.fromCssColorString(sat.color);
  sat.liveEntity = viewer.entities.add({
    id: `live-${sat.norad}`,
    name: sat.entity.name,
    show: false,
    description:
      `<p>Live SGP4 position from the GP elements of ` +
      `${sat.gp.EPOCH.slice(0, 16).replace("T", " ")} UTC.</p>`,
    position: new Cesium.CallbackProperty(
      (time, result) => liveFixedPosition(sat, time, result),
      false,
    ),
    point: {
      pixelSize: 10,
      color,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 1.5,
    },
    label: sat.entity.label?.clone(),
  });
  sat.liveOrbit = viewer.entities.add({
    id: `live-orbit-${sat.norad}`,
    show: false,
    polyline: {
      positions: new Cesium.CallbackProperty((time) => liveOrbitPositions(sat, time), false),
      width: 1.5,
      material: color.withAlpha(0.7),
      arcType: Cesium.ArcType.NONE,
    },
  });
}

function nextPassOf(sat, from) {
  // The next rise and set above the mask within NEXT_PASS_HORIZON_S, sampled
  // every NEXT_PASS_STEP_S and refined by bisection to 1 s. ">=" as in
  // visibility.find_passes. upNow: already above the mask at `from`.
  const mask = station.maskDeg;
  const at = (s) => Cesium.JulianDate.addSeconds(from, s, new Cesium.JulianDate());
  const elevation = (s) => {
    const p = liveFixedPosition(sat, at(s), liveScratch.position);
    return p ? lookAngles(p).el : -90;
  };
  const crossing = (a, b) => {
    const upA = elevation(a) >= mask;
    while (b - a > 1) {
      const mid = (a + b) / 2;
      if (elevation(mid) >= mask === upA) a = mid;
      else b = mid;
    }
    return b;
  };
  const n = NEXT_PASS_HORIZON_S / NEXT_PASS_STEP_S;
  const el = Array.from({ length: n + 1 }, (_, k) => elevation(k * NEXT_PASS_STEP_S));
  const upNow = el[0] >= mask;
  let first = 0;
  let rise;
  if (!upNow) {
    first = el.findIndex((e) => e >= mask);
    if (first < 0) return { norad: sat.norad, upNow, maxEl: Math.max(...el) };
    rise = at(crossing((first - 1) * NEXT_PASS_STEP_S, first * NEXT_PASS_STEP_S));
  }
  let last = el.findIndex((e, k) => k > first && e < mask);
  let set;
  if (last > 0) set = at(crossing((last - 1) * NEXT_PASS_STEP_S, last * NEXT_PASS_STEP_S));
  else last = n + 1;
  let maxEl = Math.max(...el.slice(first, last));
  // A LEO pass lasts minutes and peaks sharply: find its top on a finer grid.
  const fromS = Math.max(0, (first - 1) * NEXT_PASS_STEP_S);
  const toS = Math.min(NEXT_PASS_HORIZON_S, last * NEXT_PASS_STEP_S);
  if (toS - fromS <= PEAK_SEARCH_MAX_S) {
    for (let t = fromS; t <= toS; t += PEAK_STEP_S) maxEl = Math.max(maxEl, elevation(t));
  }
  return { norad: sat.norad, upNow, rise, set, maxEl };
}

function ensureNextPasses(time) {
  // Recompute for a new station, after NEXT_PASS_REFRESH_S of clock time
  // either way (the timeline can be dragged back), or once a predicted rise
  // or set has gone by.
  const passed = (jd) => jd && Cesium.JulianDate.greaterThan(time, jd);
  const stale =
    !nextPasses ||
    nextPasses.slug !== station.slug ||
    Math.abs(Cesium.JulianDate.secondsDifference(time, nextPasses.at)) >
      NEXT_PASS_REFRESH_S ||
    [...nextPasses.bySat.values()].some((p) => passed(p.rise) || passed(p.set));
  if (!stale) return;
  nextPasses = {
    slug: station.slug,
    at: time.clone(),
    bySat: new Map(
      satellites.filter((s) => s.satrec).map((s) => [s.norad, nextPassOf(s, time)]),
    ),
  };
}

function inTime(from, to) {
  const minutes = Math.round(Cesium.JulianDate.secondsDifference(to, from) / 60);
  if (minutes < 1) return "in under a minute";
  if (minutes < 60) return `in ${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `in ${h} h ${m} min` : `in ${h} h`;
}

function elementAge(sat, time) {
  const hours = Cesium.JulianDate.secondsDifference(time, sat.epochJd) / 3600;
  const size = Math.abs(hours);
  const text = size < 48 ? `${Math.round(size)} h` : `${(size / 24).toFixed(1)} d`;
  return hours >= 0 ? `elements ${text} old` : `elements from ${text} later`;
}

function updateNow(time) {
  if (station) ensureNextPasses(time);
  const deg = (x) => `${Math.round(x)}°`.replace("-", "−");
  for (const sat of satellites) {
    if (!sat.satrec) continue;
    const r = sgp4At(sat, time);
    if (!r) {
      sat.live.textContent = "SGP4 failed at this time";
      continue;
    }
    const m = Cesium.Transforms.computeTemeToPseudoFixedMatrix(time, liveScratch.teme);
    const pos = temeToFixed(r.position, m, scratch.sgp4);
    const altKm = Cesium.Cartographic.fromCartesian(pos).height / 1000;
    const v = Math.hypot(r.velocity.x, r.velocity.y, r.velocity.z); // inertial
    sat.live.textContent =
      `${Math.round(altKm).toLocaleString("en-US")} km · ${v.toFixed(3)} km/s` +
      ` · ${elementAge(sat, time)}`;
    if (!station) continue;
    const { az, el } = lookAngles(pos);
    const up = el >= station.maskDeg;
    const p = nextPasses.bySat.get(sat.norad);
    let when;
    if (up) when = p.set ? `sets ${inTime(time, p.set)}` : "up for the next 24 h";
    else if (p.rise) when = `rises ${inTime(time, p.rise)}, max ${deg(p.maxEl)}`;
    else when = "doesn't rise in the next 24 h";
    sat.sky.classList.toggle("up", up);
    sat.sky.textContent = up
      ? `${station.name}: ${deg(el)} up, azimuth ${deg(az)} · ${when}`
      : `${station.name}: ${deg(el)} · ${when}`;
  }
}

function setMode(next) {
  if (next === mode || (next === "now" && liveState !== "ready")) return;
  mode = next;
  ui.modeDay.setAttribute("aria-pressed", String(mode === "day"));
  ui.modeNow.setAttribute("aria-pressed", String(mode === "now"));
  ui.modeHint.textContent = MODE_HINTS[mode];
  ui.subtitleDay.hidden = mode !== "day";
  ui.subtitleNow.hidden = mode !== "now";
  const clock = viewer.clock;
  if (mode === "now") {
    dayClock = {
      startTime: clock.startTime.clone(),
      stopTime: clock.stopTime.clone(),
      currentTime: clock.currentTime.clone(),
      // the day's own speed, not a short pass's slowed one: slowPass is
      // cleared below, so nothing would speed the clock up again
      multiplier: slowPass ? defaultMultiplier : clock.multiplier,
      shouldAnimate: clock.shouldAnimate,
      clockRange: clock.clockRange,
      clockStep: clock.clockStep,
    };
    const now = Cesium.JulianDate.now();
    clock.startTime = Cesium.JulianDate.addHours(now, -12, new Cesium.JulianDate());
    clock.stopTime = Cesium.JulianDate.addHours(now, 12, new Cesium.JulianDate());
    clock.clockRange = Cesium.ClockRange.UNBOUNDED;
    clock.clockStep = Cesium.ClockStep.SYSTEM_CLOCK_MULTIPLIER;
    clock.multiplier = 1;
    clock.currentTime = now;
    clock.shouldAnimate = true;
    slowPass = undefined;
  } else {
    for (const [key, value] of Object.entries(dayClock)) clock[key] = value;
  }
  viewer.timeline?.zoomTo(clock.startTime, clock.stopTime);
  nextPasses = undefined;
  const tracked = trackedSatellite();
  applyVisibility();
  if (station) showStationText();
  if (tracked) follow(tracked); // the same satellite, in the new mode
  // A link with ?now opens the live mode; keep the station's #hash.
  const search = mode === "now" ? "?now" : "";
  history.replaceState(null, "", location.pathname + search + location.hash);
  lastLiveUpdate = 0;
  updateLive(clock);
}

async function prepareLiveMode() {
  try {
    const [lib, gp] = await Promise.all([
      import(SATELLITE_JS),
      fetch("gp.json").then((r) => {
        if (!r.ok) throw new Error(`gp.json: HTTP ${r.status}`);
        return r.json();
      }),
    ]);
    const byNorad = new Map(gp.map((g) => [g.NORAD_CAT_ID, g]));
    const ready = satellites.map((sat) => {
      const g = byNorad.get(sat.norad);
      if (!g) throw new Error(`gp.json has no elements for ${sat.norad}`);
      const satrec = lib.json2satrec(g);
      if (satrec.error) throw new Error(`SGP4 error ${satrec.error} for ${sat.norad}`);
      return { sat, gp: g, satrec };
    });
    sgp4lib = lib;
    for (const { sat, gp: g, satrec } of ready) {
      Object.assign(sat, {
        gp: g,
        satrec,
        epochJd: Cesium.JulianDate.fromIso8601(`${g.EPOCH}Z`),
      });
      addLiveEntities(sat);
      sat.liveEntity.viewFrom = sat.entity.viewFrom;
    }
    liveState = "ready";
    ui.modeNow.disabled = false;
    ui.modeHint.textContent = MODE_HINTS.day;
    if (new URLSearchParams(location.search).has("now")) setMode("now");
  } catch (err) {
    liveState = "unavailable";
    console.warn("Live mode unavailable:", err);
    ui.modeHint.textContent =
      "The live mode is unavailable: the SGP4 library or gp.json did not load. " +
      "The one-day view works as before.";
  }
}

// ---------- Info box ----------

// The info box renders the CZML description inside an iframe with Cesium's own
// stylesheet, which sets its own font on .cesium-infoBox-description. Restyle
// it as an Apple-style list, with the colours and font taken from style.css.
function infoBoxCss() {
  const root = getComputedStyle(document.documentElement);
  const v = (name) => root.getPropertyValue(name).trim();
  return `
    html body, html body .cesium-infoBox-description {
      margin: 0;
      font: 13px/1.38 ${root.fontFamily};
      color: ${v("--label")};
      -webkit-font-smoothing: antialiased;
    }
    html body .cesium-infoBox-description { padding: 6px 14px 12px; }
    body table { width: 100%; border-collapse: collapse; }
    body th, body td { padding: 6px 0; border: 0; background: none; }
    body th { text-align: left; font-weight: 400; color: ${v("--secondary")}; }
    body td { text-align: right; font-variant-numeric: tabular-nums; }
    body tr + tr th, body tr + tr td { border-top: 0.5px solid ${v("--hairline")}; }
    body p { margin: 4px 0; }
  `;
}

function styleInfoBox() {
  const box = viewer.infoBox;
  if (!box) return;
  box.container.querySelector(".cesium-infoBox")?.classList.add("material");
  const frame = box.frame;
  const inject = () => {
    const doc = frame.contentDocument;
    if (!doc?.head || doc.getElementById("apple-style")) return;
    const style = doc.createElement("style");
    style.id = "apple-style";
    style.textContent = infoBoxCss();
    doc.head.append(style);
  };
  frame.addEventListener("load", inject);
  inject(); // the blank frame may have loaded already
}

async function fontsReady() {
  // Cesium draws labels onto a canvas and caches them by font, so make sure
  // Inter (used where the system font isn't SF) has loaded before the CZML.
  // Never wait more than 3 s for it.
  const timeout = new Promise((resolve) => setTimeout(resolve, 3000));
  const fonts = ["500 13px Inter", "600 13px Inter"].map((f) => document.fonts.load(f));
  await Promise.race([Promise.all(fonts), timeout]).catch(() => {});
}

// ---------- Wiring ----------

styleInfoBox();
ui.fixed.addEventListener("click", () => setFrame("fixed"));
ui.inertial.addEventListener("click", () => setFrame("inertial"));
ui.freeCamera.addEventListener("click", showWholeEarth);
ui.tracks.addEventListener("change", applyVisibility);
ui.orbits.addEventListener("change", applyVisibility);
ui.showStation.addEventListener("change", applyStationVisibility);
ui.modeDay.addEventListener("click", () => setMode("day"));
ui.modeNow.addEventListener("click", () => setMode("now"));
for (const [name, button] of Object.entries(ui.models)) {
  button.addEventListener("click", () => setModel(name));
}

viewer.trackedEntityChanged.addEventListener((entity) => {
  if (entity && frame === "inertial") setFrame("fixed");
  for (const s of satellites) {
    const pressed =
      entity !== undefined && [s.entity, s.sgp4, s.liveEntity].includes(entity);
    s.button.setAttribute("aria-pressed", String(pressed));
  }
});

async function main() {
  viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(60, 15, WHOLE_EARTH_HEIGHT_M),
  });
  try {
    await fontsReady();
    const ds = await Cesium.CzmlDataSource.load("orbits.czml");
    await viewer.dataSources.add(ds); // the viewer adopts the CZML clock
    defaultMultiplier = viewer.clock.multiplier;
    const start = ds.clock.startTime;
    const day = Cesium.JulianDate.toDate(start);
    document.getElementById("window-date").textContent = `, ${day.toLocaleDateString(
      "en-GB",
      { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" },
    )}`;
    for (const entity of ds.entities.values) {
      if (entity.id.startsWith("sat-")) addSatellite(ds, entity, start);
    }
    setupStations(ds, start);
    setModel(model);
    viewer.clock.onTick.addEventListener(updateLive);
    updateLive(viewer.clock);
    ui.loading.hidden = true;
  } catch (err) {
    console.error(err);
    ui.loading.textContent =
      "Could not load orbits.czml. Serve this folder over HTTP (see README).";
    return;
  }
  await prepareLiveMode();
}

// A small, stable surface for tests/viewer and the browser console, so they
// don't depend on this file's internal names.
window.explorer = {
  viewer,
  get satellites() {
    return satellites;
  },
  get stations() {
    return stations;
  },
  get station() {
    return station;
  },
  get mode() {
    return mode;
  },
  /** "loading", "ready" or "unavailable" */
  get liveState() {
    return liveState;
  },
  setMode,
  /** Pause the clock at an ISO 8601 UTC time and refresh the panel now. */
  setTime(iso) {
    viewer.clock.shouldAnimate = false;
    viewer.clock.currentTime = Cesium.JulianDate.fromIso8601(iso);
    lastLiveUpdate = 0;
    updateLive(viewer.clock);
  },
  /** Live mode: SGP4's TEME position [km] of a satellite at an ISO time. */
  liveTemeKm(norad, iso) {
    const sat = satellites.find((s) => s.norad === norad);
    const r = sgp4At(sat, Cesium.JulianDate.fromIso8601(iso));
    return r && [r.position.x, r.position.y, r.position.z];
  },
  /** Live mode: the live marker's Earth-fixed position [m] at an ISO time. */
  liveFixedM(norad, iso) {
    const sat = satellites.find((s) => s.norad === norad);
    const p = sat.liveEntity.position.getValue(Cesium.JulianDate.fromIso8601(iso));
    return p && [p.x, p.y, p.z];
  },
  /** Live mode: each satellite's next pass over the station, from the clock. */
  nextPasses() {
    ensureNextPasses(viewer.clock.currentTime);
    const iso = (jd) => jd && Cesium.JulianDate.toIso8601(jd, 0);
    return [...nextPasses.bySat.values()].map((p) => ({
      ...p,
      rise: iso(p.rise),
      set: iso(p.set),
    }));
  },
};

main();
