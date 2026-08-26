(function () {
  var canvases = Array.prototype.slice.call(document.querySelectorAll('[data-writing-art]'));
  if (!canvases.length) return;

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var alphabet = '01KDA+-=<>[]{}:/';
  var start = performance.now();
  var frame = 0;
  var cards = [];

  function cssColor(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  function hash(x, y, seed) {
    var value = Math.sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453;
    return value - Math.floor(value);
  }

  function setup(canvas) {
    var card = canvas.closest('.writing-card');
    var state = {
      canvas: canvas,
      card: card,
      ctx: canvas.getContext('2d'),
      kind: canvas.dataset.writingArt,
      width: 1,
      height: 1,
      active: false,
      visible: true
    };

    function resize() {
      var rect = canvas.getBoundingClientRect();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      state.width = Math.max(1, rect.width);
      state.height = Math.max(1, rect.height);
      canvas.width = Math.round(state.width * dpr);
      canvas.height = Math.round(state.height * dpr);
      state.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(state, reduceMotion ? 0 : (performance.now() - start) / 1000);
    }

    card.addEventListener('mouseenter', function () { state.active = true; });
    card.addEventListener('mouseleave', function () { state.active = false; });
    card.addEventListener('focusin', function () { state.active = true; });
    card.addEventListener('focusout', function () { state.active = false; });

    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas);
    else window.addEventListener('resize', resize);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          state.visible = entry.isIntersecting;
          if (entry.isIntersecting) card.classList.add('is-visible');
        });
      }, { threshold: .12 }).observe(card);
    } else {
      card.classList.add('is-visible');
    }

    resize();
    cards.push(state);
  }

  function palette() {
    var light = document.documentElement.dataset.theme === 'light';
    return {
      text: light ? '#111713' : cssColor('--text', '#d7dbd2'),
      muted: cssColor('--muted', '#849087'),
      green: cssColor('--green', '#5b9668'),
      gold: cssColor('--gold', '#b7a16d'),
      faint: light ? 'rgba(17,23,19,.12)' : 'rgba(215,219,210,.11)'
    };
  }

  function drawCharacter(ctx, char, x, y, color, alpha, size) {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.font = '400 ' + size + 'px ui-monospace, SFMono-Regular, Consolas, monospace';
    ctx.fillText(char, x, y);
  }

  function drawKda(state, t, colors) {
    var ctx = state.ctx;
    var w = state.width;
    var h = state.height;
    var speed = state.active ? 1.28 : 1;
    var time = t * speed;
    var cx = w * .5;
    var cy = h * .47;
    var radius = Math.min(w * .31, h * .38) * (1 + Math.sin(time * .42) * .004);
    var cell = Math.max(8.6, Math.min(10.8, radius / 14));
    var accent = colors.green;
    var lightAngle = -.72 + Math.sin(time * .16) * .42;
    var lightX = Math.cos(lightAngle) * -.72;
    var lightY = -.42 + Math.sin(time * .11) * .16;
    var lightZ = Math.sin(lightAngle) * .72 + .35;
    var lightLength = Math.sqrt(lightX * lightX + lightY * lightY + lightZ * lightZ);

    lightX /= lightLength;
    lightY /= lightLength;
    lightZ /= lightLength;

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    function rotate3d(point, ax, ay) {
      var cosY = Math.cos(ay);
      var sinY = Math.sin(ay);
      var x1 = point.x * cosY + point.z * sinY;
      var z1 = -point.x * sinY + point.z * cosY;
      var cosX = Math.cos(ax);
      var sinX = Math.sin(ax);
      return {
        x: x1,
        y: point.y * cosX - z1 * sinX,
        z: point.y * sinX + z1 * cosX
      };
    }

    function digit(seed, phase) {
      return hash(seed, phase, 19) > .5 ? '1' : '0';
    }

    /* The sphere stays geometrically still. Its surface behaves like a quiet
       data field: illumination, density, and several currents move separately. */
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius + cell * .45, 0, Math.PI * 2);
    ctx.clip();

    var phase = Math.floor(time * .34);
    var gridRadius = Math.ceil(radius / cell);
    for (var row = -gridRadius; row <= gridRadius; row += 1) {
      for (var column = -gridRadius; column <= gridRadius; column += 1) {
        var seed = (row + gridRadius) * 97 + column + gridRadius;
        var jitterX = (hash(column, row, 3) - .5) * cell * .22;
        var jitterY = (hash(column, row, 7) - .5) * cell * .22;
        var px = column * cell + jitterX;
        var py = row * cell + jitterY;
        var nx = px / radius;
        var ny = py / radius;
        var radial = nx * nx + ny * ny;
        if (radial >= 1) continue;

        var nz = Math.sqrt(1 - radial);
        var longitude = Math.atan2(nx, nz);
        var latitude = Math.asin(ny);
        var lambert = nx * lightX + ny * lightY + nz * lightZ;
        var terminator = Math.max(0, Math.min(1, (lambert + .2) / .9));
        var rim = Math.pow(radial, 2.2);
        var current = .5 + .5 * Math.sin(longitude * 4.2 + latitude * 7.4 - time * .74);
        var probability = .27 + terminator * .51 + current * .12 + rim * .1;
        if (hash(column, row, 43) > probability) continue;

        var alpha = .18 + terminator * .7 + current * .08;
        var useAccent = current > .84 && terminator > .32;
        drawCharacter(
          ctx,
          digit(seed, phase + Math.floor(longitude * 3)),
          cx + px,
          cy + py,
          useAccent ? accent : colors.text,
          alpha,
          cell * .88
        );
      }
    }

    var currents = [
      { ax: .18, ay: .08, speed: .19, count: 25 },
      { ax: 1.02, ay: -.34, speed: -.14, count: 21 },
      { ax: -.7, ay: .52, speed: .11, count: 19 },
      { ax: .48, ay: 1.1, speed: -.09, count: 17 }
    ];

    currents.forEach(function (current, currentIndex) {
      for (var item = 0; item < current.count; item += 1) {
        var theta = Math.PI * 2 * (item / current.count + time * current.speed);
        var point = rotate3d({ x: Math.cos(theta), y: Math.sin(theta), z: 0 }, current.ax, current.ay);
        var trail = .5 + .5 * Math.sin(theta * 2 - time * .45 + currentIndex);
        var front = .25 + Math.max(0, point.z) * .75;
        drawCharacter(
          ctx,
          digit(item + currentIndex * 41, phase + currentIndex),
          cx + point.x * radius * .97,
          cy + point.y * radius * .97,
          currentIndex === 0 || trail > .84 ? accent : colors.text,
          front * (.32 + trail * .5),
          cell * (point.z > 0 ? .88 : .68)
        );
      }
    });
    ctx.restore();

    /* Several loose eddies shed information from different parts of the edge.
       Their unequal radii keep the halo from reading as another perfect ring. */
    var streamCount = 42;
    for (var streamIndex = 0; streamIndex < streamCount; streamIndex += 1) {
      var streamSeed = hash(streamIndex, 13, 29);
      var progress = (streamSeed + time * (.018 + hash(streamIndex, 5, 11) * .022)) % 1;
      var sourceAngle = hash(streamIndex, 19, 31) * Math.PI * 2;
      var direction = streamIndex % 2 ? 1 : -1;
      var curl = (.42 + hash(streamIndex, 7, 37) * .9) * direction;
      var streamAngle = sourceAngle + curl * progress + Math.sin(progress * Math.PI * 2 + streamSeed * 9) * .1;
      var excursion = .13 + hash(streamIndex, 23, 41) * .48;
      var streamRadius = radius * (1.01 + Math.sin(Math.PI * progress) * excursion);
      var streamX = cx + Math.cos(streamAngle) * streamRadius;
      var streamY = cy + Math.sin(streamAngle) * streamRadius * (.82 + hash(streamIndex, 3, 47) * .22);
      var fade = Math.pow(Math.sin(Math.PI * progress), .7);
      drawCharacter(
        ctx,
        digit(streamIndex, phase),
        streamX,
        streamY,
        streamIndex % 5 === 0 ? accent : colors.text,
        fade * (.16 + hash(streamIndex, 2, 17) * .42),
        cell * (.55 + hash(streamIndex, 5, 23) * .4)
      );
    }
  }

  function drawCrawl(state, t, colors) {
    var ctx = state.ctx;
    var w = state.width;
    var h = state.height;
    var speed = state.active ? .155 : .1;
    var top = h * .035;
    var funnelBottom = h * .61;
    var center = w * .5;
    var flowHeight = funnelBottom - top;
    var glyphs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:/[]{}';
    var stages = [0, .27, .53, .77, 1];
    var widths = [.92, .65, .43, .25, .105];
    var particleCount = Math.max(165, Math.round(w * .56));

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    function flowWidth(progress) {
      for (var stage = 0; stage < stages.length - 1; stage += 1) {
        if (progress <= stages[stage + 1]) {
          var local = (progress - stages[stage]) / (stages[stage + 1] - stages[stage]);
          return widths[stage] + (widths[stage + 1] - widths[stage]) * local;
        }
      }
      return widths[widths.length - 1];
    }

    for (var particle = 0; particle < particleCount; particle += 1) {
      var seed = hash(particle, 2, 5);
      var progress = (seed + t * speed) % 1;
      var quality = hash(particle, 7, 11);
      var rawOffset = hash(particle, 13, 17) - .5;
      var rejection = 1.1;

      if (quality < .45) rejection = stages[1];
      else if (quality < .69) rejection = stages[2];
      else if (quality < .84) rejection = stages[3];
      else if (quality < .91) rejection = stages[4];

      var activeProgress = Math.min(progress, rejection);
      var widthFactor = flowWidth(activeProgress);
      var baseX = center + rawOffset * w * widthFactor;
      var x = baseX;
      var y = top + activeProgress * flowHeight;
      var alpha;
      var color;
      var size = Math.max(5.4, Math.min(6.8, w / 51));

      if (progress > rejection) {
        var rejectProgress = (progress - rejection) / .11;
        if (rejectProgress > 1) continue;
        var direction = hash(particle, 19, 23) > .5 ? 1 : -1;
        x = baseX + direction * rejectProgress * w * (.055 + hash(particle, 29, 31) * .065);
        y += rejectProgress * h * .035;
        alpha = (1 - rejectProgress) * (.13 + quality * .2);
        color = colors.muted;
        size *= .78 + (1 - rejectProgress) * .16;
      } else {
        var structure = Math.max(0, (progress - stages[3]) / (1 - stages[3]));
        if (structure > 0) {
          var slot = Math.floor(hash(particle, 37, 41) * 5);
          var targetX = center + (slot - 2) * Math.max(4, w * .012);
          x += (targetX - x) * structure * structure;
        }
        alpha = .14 + quality * .36 + progress * .2;
        color = progress > stages[2] ? colors.green : colors.text;
      }

      var charIndex = Math.floor(hash(particle, Math.floor(t * .75), 47) * glyphs.length);
      drawCharacter(ctx, glyphs.charAt(charIndex), x, y, color, alpha, size);
    }

    /* The final narrow stream resolves into ordinary prose. The crop stays
       fixed while the paragraph moves upward forever, like a dataset reader. */
    var textTop = h * .665;
    var textBottom = h * .965;
    var textLeft = w * .09;
    var textWidth = w * .82;
    var fontSize = Math.max(5.6, Math.min(7.2, w / 47));
    var lineHeight = fontSize * 1.65;
    var lines = [
      'The web is large, repetitive, and unevenly written.',
      'Useful language sits beside menus, spam, and boilerplate.',
      'Each pass removes a different kind of noise from the crawl.',
      'Documents become passages; passages become clean examples.',
      'Code, science, dialogue, and explanation survive the filters.',
      'The remaining text is packed into a dataset for training.',
      'What begins as raw pages ends as readable language.'
    ];
    var cycleHeight = lines.length * lineHeight;
    var scroll = (t * (state.active ? 6.2 : 4.1)) % cycleHeight;

    ctx.save();
    ctx.beginPath();
    ctx.rect(textLeft, textTop, textWidth, textBottom - textTop);
    ctx.clip();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '400 ' + fontSize + 'px ui-monospace, SFMono-Regular, Consolas, monospace';

    for (var repeat = -1; repeat <= 1; repeat += 1) {
      for (var lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
        var lineY = textTop + lineIndex * lineHeight - scroll + repeat * cycleHeight;
        var edgeFade = Math.min(1, (lineY - textTop) / (lineHeight * 1.3), (textBottom - lineY) / (lineHeight * 1.3));
        if (edgeFade <= 0) continue;
        ctx.globalAlpha = (.25 + lineIndex % 3 * .07) * edgeFade;
        ctx.fillStyle = lineIndex % 3 === 1 ? colors.green : colors.text;
        ctx.fillText(lines[lineIndex], textLeft, lineY);
      }
    }
    ctx.restore();
  }

  function draw(state, t) {
    var ctx = state.ctx;
    var colors = palette();
    ctx.clearRect(0, 0, state.width, state.height);
    if (state.kind === 'kda') drawKda(state, t, colors);
    else drawCrawl(state, t, colors);
    ctx.globalAlpha = 1;
  }

  function loop(now) {
    var t = (now - start) / 1000;
    cards.forEach(function (state) {
      if (state.visible) draw(state, t);
    });
    frame = window.requestAnimationFrame(loop);
  }

  canvases.forEach(setup);

  if (!reduceMotion) frame = window.requestAnimationFrame(loop);

  document.addEventListener('visibilitychange', function () {
    if (document.hidden && frame) {
      window.cancelAnimationFrame(frame);
      frame = 0;
    } else if (!document.hidden && !reduceMotion && !frame) {
      frame = window.requestAnimationFrame(loop);
    }
  });
})();
