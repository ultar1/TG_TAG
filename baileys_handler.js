const makeWASocket = require('@whiskeysockets/baileys').default;
const {
  useMultiFileAuthState,
  makeCacheableSignalKeyStore
} = require('@whiskeysockets/baileys');
const pino = require('pino');
const fs = require('fs');
const readline = require('readline');

// Command-line arguments for the bot
const args = process.argv.slice(2);
const sessionName = args[0] || 'Ultar_Session';

// Logger
const logger = pino({
  level: 'silent'
});

async function connectToWhatsApp() {
  const {
    state,
    saveCreds
  } = await useMultiFileAuthState(sessionName);

  const sock = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    logger,
    browser: ['Chrome (Linux)', '', ''],
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', async (update) => {
    const {
      connection,
      lastDisconnect,
      isNewLogin
    } = update;
    if (isNewLogin) {
      const code = await sock.requestPairingCode(args[1] || '91xxxxxxxxxx'); // You'll pass the phone number here
      console.log(`PAIRING_CODE: ${code}`);
      return;
    }
    if (connection === 'close') {
      const shouldReconnect = lastDisconnect.error?.output?.statusCode !== 401;
      console.log('connection closed due to ', lastDisconnect.error, ', reconnecting ', shouldReconnect);
      if (shouldReconnect) {
        connectToWhatsApp();
      } else {
        console.log("CONNECTION_STATUS: BANNED");
      }
    } else if (connection === 'open') {
      console.log("CONNECTION_STATUS: CONNECTED");
    }
  });
}

connectToWhatsApp();
