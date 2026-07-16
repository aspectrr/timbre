import { createSignal } from "solid-js";

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

export default function Guide() {
  return (
    <div>
      <div class="toc">
        <a href="#g1">1 · Get your email</a>
        <a href="#g2">2 · Train</a>
        <a href="#g3">3 · Use your model</a>
      </div>

      <div class="card" id="g1">
        <h2>1 · Get your email as a file</h2>
        <p style={{ color: "var(--mut)", margin: "0 0 6px" }}>
          This app learns from emails <b style={{ color: "var(--ink)" }}>you</b> wrote.
          Export your "Sent" mail into a file (called an mbox) and upload it. Nothing is
          deleted from your account — exporting just makes a copy.
        </p>

        <details open>
          <summary>Gmail — via Google Takeout <span class="pill">easiest</span></summary>
          <div class="inner">
            <ol class="steps">
              <li>Go to <a href="https://takeout.google.com" target="_blank">takeout.google.com</a> and sign in.</li>
              <li>Click <b>"Deselect all"</b> at the top of the product list.</li>
              <li>Scroll to <b>Mail</b> and tick its checkbox.</li>
              <li>Click <b>"All Mail data included"</b> → untick "All Mail" → tick just <b>"Sent"</b>.</li>
              <li>Scroll to the bottom → <b>"Next step"</b>.</li>
              <li>Leave it as <b>"Send download link via email"</b>, <b>"Export once"</b>, <b>.zip</b> → click <b>"Create export"</b>.</li>
              <li>Wait for Google's email (minutes to hours). Download the <code>.zip</code>.</li>
              <li>Unzip it. Inside you'll find <code>Sent.mbox</code>.</li>
              <li>Upload that <code>.mbox</code> file here.</li>
            </ol>
            <div class="note">Exporting is a copy — your Gmail account, labels, and messages are untouched.</div>
          </div>
        </details>

        <details>
          <summary>iCloud — via Thunderbird</summary>
          <div class="inner">
            <p style={{ color: "var(--mut)" }}>iCloud has no "download your mail" button, so we use the free Thunderbird app.</p>
            <ol class="steps">
              <li>Install <a href="https://www.thunderbird.net" target="_blank">Thunderbird</a> (free).</li>
              <li>Make an <b>app-specific password</b>: <a href="https://account.apple.com" target="_blank">account.apple.com</a> → App-Specific Passwords → generate one labeled "Thunderbird". Copy it.</li>
              <li>Open Thunderbird → add your mail account. Enter your name, full <code>@icloud.com</code> address, and the <b>app-specific password</b>.</li>
              <li>If it asks for server settings: <b>IMAP</b> <code>imap.mail.me.com</code>, <b>port</b> 993, <b>SSL</b> on.</li>
              <li>Let folders sync (can take a while).</li>
              <li>Install the <b>ImportExportTools NG</b> add-on (Add-ons and Themes → search → Add).</li>
              <li>Right-click <b>Sent</b> → ImportExportTools NG → <b>Export folder</b> → save.</li>
              <li>Upload the <code>.mbox</code> here.</li>
            </ol>
          </div>
        </details>

        <details>
          <summary>Outlook, Yahoo, or other IMAP — via Thunderbird</summary>
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
          <summary>What about notes, docs, or other writing?</summary>
          <div class="inner">
            <p style={{ color: "var(--mut)" }}>Save them as <code>.txt</code> or <code>.md</code> and upload those. Each file becomes a writing sample. Good sources: sent email, personal notes, blog posts, Slack exports. Aim for 50–100+ messages for a recognizable style.</p>
          </div>
        </details>
      </div>

      <div class="card" id="g2">
        <h2>2 · Train your model</h2>
        <ol class="steps">
          <li>Enter the email address <b>you send from</b>.</li>
          <li>Pick a teacher model — <span class="acc">Claude Opus</span> gives the best result; Flash is cheaper.</li>
          <li>Upload your file(s).</li>
          <li>Click <b>Train my style model</b>.</li>
        </ol>
        <div class="note">Takes about 15 minutes. You'll see live progress. Come back anytime — the job is durable and resumes on its own.</div>
      </div>

      <div class="card" id="g3">
        <h2>3 · Run your model on your computer</h2>
        <p style={{ color: "var(--mut)", margin: "0 0 8px" }}>When training finishes, download <b>both</b> files: <code>adapter.gguf</code> and <code>Modelfile</code>. Then:</p>
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
      </div>
    </div>
  );
}
