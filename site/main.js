// Interactivity for the lilbee site: the mega-nav menus, the install-method
// tabs, the tutorial reel tabs, hash deep links that activate a tab, the
// per-method "details" disclosures, and the copy-to-clipboard buttons.

(function () {
  'use strict';

  // tab element -> the select() of the tablist that owns it, so deep links
  // can activate a tab without synthesizing clicks.
  var tabSelects = new Map();

  /** Wire every [role="tablist"] on the page (install-method tabs, tutorial reel,
      anything else with the same ARIA shape). Click selection plus left/right
      arrow-key navigation. */
  function initTablists() {
    var tablists = document.querySelectorAll('[role="tablist"]');
    Array.prototype.forEach.call(tablists, function (tablist) {
      var tabs = Array.prototype.slice.call(tablist.querySelectorAll('[role="tab"]'));

      function select(tab) {
        tabs.forEach(function (candidate) {
          var active = candidate === tab;
          candidate.setAttribute('aria-selected', active ? 'true' : 'false');
          candidate.tabIndex = active ? 0 : -1;
          var pane = document.getElementById(candidate.getAttribute('aria-controls'));
          if (pane) pane.hidden = !active;
        });
      }

      tabs.forEach(function (tab) { tabSelects.set(tab, select); });

      tablist.addEventListener('click', function (event) {
        var tab = event.target.closest('[role="tab"]');
        if (!tab) return;
        select(tab);
        tab.focus();
      });

      tablist.addEventListener('keydown', function (event) {
        var index = tabs.indexOf(document.activeElement);
        if (index < 0) return;
        var step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
        if (step === 0) return;
        event.preventDefault();
        var next = tabs[(index + step + tabs.length) % tabs.length];
        select(next);
        next.focus();
      });
    });
  }

  /** Mega-nav dropdowns: click to toggle, Escape or an outside click closes,
      up/down arrows move through the items, Tab moves on and closes. */
  function initMenus() {
    var menus = Array.prototype.slice.call(document.querySelectorAll('.menu'));

    function close(menu) {
      menu.classList.remove('open');
      menu.querySelector('.menu-btn').setAttribute('aria-expanded', 'false');
    }
    function closeAll() { menus.forEach(close); }

    menus.forEach(function (menu) {
      var button = menu.querySelector('.menu-btn');
      var links = Array.prototype.slice.call(menu.querySelectorAll('.menu-panel a'));

      button.addEventListener('click', function () {
        var willOpen = !menu.classList.contains('open');
        closeAll();
        if (willOpen) {
          menu.classList.add('open');
          button.setAttribute('aria-expanded', 'true');
        }
      });

      menu.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
          close(menu);
          button.focus();
          return;
        }
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        event.preventDefault();
        if (!menu.classList.contains('open')) {
          menu.classList.add('open');
          button.setAttribute('aria-expanded', 'true');
          links[event.key === 'ArrowDown' ? 0 : links.length - 1].focus();
          return;
        }
        var index = links.indexOf(document.activeElement);
        var step = event.key === 'ArrowDown' ? 1 : -1;
        var next = links[(index + step + links.length) % links.length] || links[0];
        next.focus();
      });

      menu.addEventListener('focusout', function (event) {
        if (event.relatedTarget && menu.contains(event.relatedTarget)) return;
        close(menu);
      });
    });

    document.addEventListener('click', function (event) {
      if (!event.target.closest('.menu')) closeAll();
    });
  }

  function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /** Map a "#demo-chat" / "#install-brew" hash to the tab it names. */
  function tabForHash(hash) {
    var match = /^#(demo|install)-(.+)$/.exec(hash || '');
    if (!match) return null;
    var id = match[1] === 'demo' ? 'tutorial-tab-' + match[2] : 'tab-' + match[2];
    return document.getElementById(id);
  }

  /** Activate the tab a hash names and scroll its section into view. */
  function activateHash(hash, smooth) {
    var tab = tabForHash(hash);
    if (!tab) return false;
    var select = tabSelects.get(tab);
    if (select) select(tab);
    var section = tab.closest('section');
    if (section) section.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
    return true;
  }

  /** Mega-nav deep links: when the link points at a tab on this page, activate
      it in place; otherwise let the browser carry the hash to the other page,
      where the load-time activateHash picks it up. */
  function initDeepLinks() {
    document.addEventListener('click', function (event) {
      var link = event.target.closest('.menu-panel a[href*="/#"]');
      if (!link) return;
      var url = new URL(link.href, location.href);
      if (url.pathname !== location.pathname || !tabForHash(url.hash)) return;
      event.preventDefault();
      activateHash(url.hash, !prefersReducedMotion());
      history.pushState(null, '', url.hash);
      var menu = link.closest('.menu');
      if (menu) {
        menu.classList.remove('open');
        menu.querySelector('.menu-btn').setAttribute('aria-expanded', 'false');
      }
    });

    activateHash(location.hash, false);
    window.addEventListener('hashchange', function () {
      activateHash(location.hash, !prefersReducedMotion());
    });
  }

  /** Toggle the per-method "details" notes when their disclosure button is clicked. */
  function initDetailsToggles() {
    document.addEventListener('click', function (event) {
      var toggle = event.target.closest('.notes-toggle');
      if (!toggle) return;
      var open = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', open ? 'false' : 'true');
      var body = document.getElementById(toggle.getAttribute('aria-controls'));
      if (body) body.hidden = open;
    });
  }

  /** Copy a command to the clipboard when its [ COPY ] button is clicked. */
  function initCopyButtons() {
    document.addEventListener('click', function (event) {
      var button = event.target.closest('.copy');
      if (!button) return;
      var row = button.closest('.install');
      var command = row ? row.querySelector('.cmd') : null;
      copyText(command ? command.textContent : '', function () { flashCopied(button); });
    });
  }

  function copyText(text, done) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, done);
      return;
    }
    var area = document.createElement('textarea');
    area.value = text;
    document.body.appendChild(area);
    area.select();
    try { document.execCommand('copy'); } catch (error) { /* clipboard unavailable */ }
    document.body.removeChild(area);
    done();
  }

  function flashCopied(button) {
    if (button.textContent.indexOf('COPIED') !== -1) return;
    var original = button.textContent;
    button.textContent = '[ COPIED ]';
    setTimeout(function () { button.textContent = original; }, 1200);
  }

  initTablists();
  initMenus();
  initDeepLinks();
  initDetailsToggles();
  initCopyButtons();
})();
