import { createSignal } from "solid-js";
import { loadKey, mintKey, apiBase } from "./api.js";

function CopyBtn(props) {
  const [ok, setOk] = createSignal(false);
  const copy = () => {
    navigator.clipboard.writeText(props.text).then(() => {
      setOk(true);
      setTimeout(() => setOk(false), 1400);
    });
  };
  return (
    <button class={`copy ${ok() ? "ok" : ""}`} onClick={copy}>
      {ok() ? "copied" : "copy"}
    </button>
  );
}

function Cmd(props) {
  return (
    <div class="cmd">
      <code>{props.text}</code>
      <CopyBtn text={props.text} />
    </div>
  );
}

// a config block with a label + copyable content, rendered like .cmd but
// allowing multi-line (pre) content for JSON/TOML configs.
function ConfigBlock(props) {
  return (
    <div class="cmd" style={{ "align-items": "flex-start" }}>
      <pre style={{ margin: "0", "white-space": "pre-wrap", "word-break": "break-all", font: "inherit" }}><code>{props.text}</code></pre>
      <CopyBtn text={props.text} />
    </div>
  );
}

// ── agent path: key + MCP config + skill install ─────────────────────────
// The key is auto-minted on first visit (App onMount) and lives in localStorage.
// The agent gets the SAME key (pasted into its MCP config), so jobs it starts
// show up here — the web UI and the agent share one key.
function AgentSection() {
  const [key, setKey] = createSignal(loadKey() || "");
  const [err, setErr] = createSignal("");

  const generate = async () => {
    setErr("");
    try { setKey(await mintKey("my-agent")); }
    catch (e) { setErr(e.message || "could not reach the server"); }
  };

  // Trailing slash required: the streamable-HTTP transport establishes a
  // session at /mcp/. Without it, /mcp 307-redirects to /mcp/ and the redirect
  // leg drops the Authorization header, breaking auth on subsequent calls.
  const url = apiBase() + "/mcp/";
  const k = () => key() || "<paste your key>";

  // per-client MCP configs (auto-filled when a key exists)
  const piConfig = JSON.stringify({
    mcpServers: { timbre: { url, auth: "bearer", bearerToken: k() } },
  }, null, 2);

  const claudeDesktopConfig = JSON.stringify({
    mcpServers: { timbre: { type: "http", url, headers: { Authorization: `Bearer ${k()}` } } },
  }, null, 2);

  const codexConfig =
    `[mcp_servers.timbre]\n` +
    `url = "${url}"\n` +
    `bearer_token_env_var = "TIMBRE_API_KEY"`;

  const opencodeConfig = JSON.stringify({
    $schema: "https://opencode.ai/config.json",
    mcp: { timbre: { type: "remote", url, enabled: true, headers: { Authorization: `Bearer ${k()}` } } },
  }, null, 2);

  // Claude Code is a one-liner CLI command (shown open); the rest are config files.
  const claudeCodeCmd = `claude mcp add --transport http timbre ${url} \\\n  --header "Authorization: Bearer ${k()}"`;

  // skill install via npx skills from the GitHub repo
  const skillsCmd = "npx skills add aspectrr/timbre --skill timbre -g -y";

  return (
    <div class="agent-panel">
      <p class="body-text">
        The easiest path: hand this to your AI agent. It interviews you for where
        your writing lives, gathers it, and starts the run — then you come back
        <b> here</b> to watch it train. You barely do anything.
      </p>

      <div class="field" style={{ "margin-top": "40px" }}>
        <label>1 · Your API key</label>
        {key() ? (
          <div class="key-reveal">
            <span>This key links this page to your agent's runs</span>
            <code>{key()}</code>
            <div style={{ display: "flex", gap: "8px", "margin-top": "4px" }}>
              <CopyBtn text={key()} />
              <button class="copy" onClick={generate}>generate new</button>
            </div>
          </div>
        ) : (
          <>
            <button class="pick-btn" onClick={generate} style={{ width: "auto" }}>Generate API key</button>
            {err() && <div class="errbox" style={{ "margin-top": "16px" }}>{err()}</div>}
          </>
        )}
      </div>

      <div class="field">
        <label>2 · Connect your agent's MCP server</label>
        <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%" }}>
          Claude Code is shown by default. Others are below it.
        </p>

        <details open>
          <summary>Claude Code</summary>
          <div class="inner">
            <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%" }}>
              Run this in your terminal (Claude Code installs the server to your config):
            </p>
            <ConfigBlock text={claudeCodeCmd} />
          </div>
        </details>

        <details>
          <summary>pi</summary>
          <div class="inner">
            <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%" }}>
              Add to <code>~/.pi/agent/mcp.json</code>:
            </p>
            <ConfigBlock text={piConfig} />
          </div>
        </details>

        <details>
          <summary>Claude Desktop</summary>
          <div class="inner">
            <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%" }}>
              Add to <code>claude_desktop_config.json</code>
              (Mac: <code>~/Library/Application&nbsp;Support/Claude/</code>;
              Windows: <code>%APPDATA%\Claude\</code>):
            </p>
            <ConfigBlock text={claudeDesktopConfig} />
          </div>
        </details>

        <details>
          <summary>Codex</summary>
          <div class="inner">
            <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%" }}>
              Add to <code>~/.codex/config.toml</code>, then put your key in an env var:
            </p>
            <ConfigBlock text={codexConfig} />
            <Cmd text={`export TIMBRE_API_KEY="${k()}"`} />
          </div>
        </details>

        <details>
          <summary>OpenCode</summary>
          <div class="inner">
            <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%" }}>
              Add to <code>opencode.json</code>:
            </p>
            <ConfigBlock text={opencodeConfig} />
          </div>
        </details>
      </div>

      <div class="field">
        <label>3 · Install the skill</label>
        <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%" }}>
          The <a href="https://github.com/aspectrr/timbre/blob/main/skills/timbre/SKILL.md" target="_blank">Timbre skill</a> teaches your agent how to run a job.
          Install it from the repo with the skills CLI (auto-detects your agent;
          add <code>-a claude-code</code>, <code>-a codex</code>, or <code>-a opencode</code> to target one):
        </p>
        <Cmd text={skillsCmd} />
        <p class="body-text" style={{ "font-size": "var(--text-caption)", "max-width": "100%", "margin-top": "12px" }}>
          No CLI? <a href="https://agentskills.io" target="_blank">Agent Skills</a> is an open
          standard — drop <code>SKILL.md</code> into your agent's skills folder
          (<code>~/.claude/skills/timbre/</code>, <code>~/.codex/skills/timbre/</code>, etc.).
        </p>
      </div>

      <div class="field">
        <label>4 · Say the word</label>
        <p class="body-text">
          Tell your agent something like:
        </p>
        <Cmd text="Train my writing style." />
        <p class="body-text" style={{ "margin-top": "16px" }}>
          It asks where your writing lives (Gmail, Obsidian, docs…), walks you
          through pulling it, and starts. Then open the <b>Train</b> tab here —
          your run appears automatically because you share the same key.
        </p>
      </div>

      <p class="note">
        Prefer to do it by hand? The steps below cover exporting your writing,
        uploading, and running the model locally.
      </p>
    </div>
  );
}

export default function Guide() {
  return (
    <div>
      <div class="toc">
        <a href="#g0">00 · Agent</a>
        <a href="#g1">01 · Source</a>
        <a href="#g2">02 · Train</a>
        <a href="#g3">03 · Run</a>
      </div>

      {/* ── 00 · LET YOUR AGENT DO IT ── */}
      <section id="g0" style={{ "padding-top": "0" }}>
        <p class="label">Easiest path</p>
        <h2 class="section-title">Let your agent do it</h2>
        <AgentSection />
      </section>

      {/* ── 01 · GET YOUR EMAIL ── */}
      <section id="g1" style={{ "padding-top": "0" }}>
        <p class="label">Step One</p>
        <h2 class="section-title">Source your writing</h2>
        <p class="body-text">
          This app learns from things <b>you</b> wrote. Export your Sent mail into
          a file (an mbox) and upload it, or use notes, docs, and other writing.
          Exporting is a copy — nothing is deleted from your account.
        </p>

        <div style={{ "margin-top": "48px" }}>
          <details open>
            <summary>Gmail — Google Takeout</summary>
            <div class="inner">
              <ol class="steps">
                <li>Go to <a href="https://takeout.google.com" target="_blank">takeout.google.com</a> and sign in.</li>
                <li>Click <b>"Deselect all"</b> at the top of the product list.</li>
                <li>Scroll to <b>Mail</b> and tick its checkbox.</li>
                <li>Click <b>"All Mail data included"</b> → untick "All Mail" → tick just <b>"Sent"</b>.</li>
                <li>Scroll to the bottom → <b>"Next step"</b>.</li>
                <li>Leave it as <b>"Send download link via email"</b>, <b>"Export once"</b>, <b>.zip</b> → <b>"Create export"</b>.</li>
                <li>Wait for Google's email (minutes to hours). Download the <code>.zip</code>.</li>
                <li>Unzip it. Inside you'll find <code>Sent.mbox</code>.</li>
                <li>Upload that <code>.mbox</code> here.</li>
              </ol>
              <p class="note">A copy — your Gmail account, labels, and messages are untouched.</p>
            </div>
          </details>

          <details>
            <summary>iCloud — via Thunderbird</summary>
            <div class="inner">
              <p class="body-text">iCloud has no "download your mail" button, so we use the free Thunderbird app.</p>
              <ol class="steps">
                <li>Install <a href="https://www.thunderbird.net" target="_blank">Thunderbird</a> (free).</li>
                <li>Make an <b>app-specific password</b>: <a href="https://account.apple.com" target="_blank">account.apple.com</a> → App-Specific Passwords → generate one labeled "Thunderbird". Copy it.</li>
                <li>Open Thunderbird → add your mail account. Enter your name, full <code>@icloud.com</code> address, and the <b>app-specific password</b>.</li>
                <li>If it asks for server settings: <b>IMAP</b> <code>imap.mail.me.com</code>, <b>port</b> 993, <b>SSL</b> on.</li>
                <li>Let folders sync (can take a while).</li>
                <li>Install <b>ImportExportTools NG</b> (Add-ons → search → Add).</li>
                <li>Right-click <b>Sent</b> → ImportExportTools NG → <b>Export folder</b> → save.</li>
                <li>Upload the <code>.mbox</code> here.</li>
              </ol>
            </div>
          </details>

          <details>
            <summary>Outlook, Yahoo, or other IMAP</summary>
            <div class="inner">
              <ol class="steps">
                <li>Install <a href="https://www.thunderbird.net" target="_blank">Thunderbird</a> (free).</li>
                <li>Most providers require an <b>app password</b> now. Generate one in your provider's security settings.</li>
                <li>Add your account in Thunderbird using that app password.</li>
                <li>Install <b>ImportExportTools NG</b>, right-click <b>Sent</b> → Export folder → save the <code>.mbox</code>.</li>
                <li>Upload the <code>.mbox</code> here.</li>
              </ol>
            </div>
          </details>

          <details>
            <summary>Notes, Obsidian, docs, or other writing</summary>
            <div class="inner">
              <p class="body-text">
                Anything <b>you</b> wrote works. Accepted formats: <code>.txt</code>,{" "}
                <code>.md</code> (Obsidian — frontmatter and links are cleaned
                automatically), <code>.docx</code>, and <code>.pdf</code>. Good
                sources: sent email, personal notes, blog posts, journal entries.
                Aim for 50–100+ samples for a recognizable style.
              </p>
            </div>
          </details>
        </div>
      </section>

      {/* ── 02 · TRAIN ── */}
      <section id="g2">
        <p class="label">Step Two</p>
        <h2 class="section-title">Train your model</h2>
        <ol class="steps">
          <li>Enter the email address <b>you send from</b> — only needed for <code>.mbox</code>/<code>.eml</code>.</li>
          <li>Pick a teacher model — <b>Opus</b> gives the best result; <b>Flash</b> is cheaper.</li>
          <li>Upload your file(s).</li>
          <li>Click <b>Train</b>.</li>
        </ol>
        <p class="note">Takes about 15 minutes. You'll see live progress. Come back anytime — the job is durable and resumes on its own.</p>
      </section>

      {/* ── 03 · RUN ── */}
      <section id="g3">
        <p class="label">Step Three</p>
        <h2 class="section-title">Run it locally</h2>
        <p class="body-text">
          When training finishes, download <b>both</b> files: <code>adapter.gguf</code> and <code>Modelfile</code>.
        </p>
        <ol class="steps">
          <li>Install <a href="https://ollama.com" target="_blank">Ollama</a> (free).</li>
          <li>Open <b>Terminal</b> (Mac: Cmd + Space → "Terminal").</li>
          <li>Download the base model once:
            <Cmd text="ollama pull llama3.2:3b" />
          </li>
          <li>Put <code>adapter.gguf</code> and <code>Modelfile</code> in one folder (e.g. <code>~/Desktop/my-model</code>).</li>
          <li>Build your model:
            <Cmd text="cd ~/Desktop/my-model" />
            <Cmd text="ollama create my-style -f Modelfile" />
          </li>
          <li>Run it:
            <Cmd text="ollama run my-style" />
          </li>
          <li>Type a request — e.g. <i>"write a quick follow-up to a client who hasn't replied"</i> — and it writes in your voice.</li>
        </ol>
      </section>
    </div>
  );
}
