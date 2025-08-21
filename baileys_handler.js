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
// The phone number is the second argument
const phoneNumber = args[1] ? args[1].replace('+', '') : null;

// Logger
const logger = pino({
  level: 'info' // Change log level to 'info' for more details
});

if (!phoneNumber) {
    console.error("ERROR: Phone number not provided.");
    process.exit(1);
}

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
    
    logger.info(`Connection update received: ${JSON.stringify(update)}`);

    if (isNewLogin) {
      try {
        const code = await sock.requestPairingCode(phoneNumber);
        console.log(`PAIRING_CODE: ${code}`);
        logger.info("Successfully generated pairing code.");
      } catch (e) {
        console.error(`ERROR: Failed to generate pairing code. ${e.message}`);
        logger.error(`Failed to request pairing code: ${e}`);
        // Exit the process so Python can detect the failure
        process.exit(1);
      }
      return;
    }
    
    if (connection === 'close') {
      const shouldReconnect = lastDisconnect.error?.output?.statusCode !== 401;
      logger.info(`Connection closed. Should reconnect: ${shouldReconnect}`);
      if (shouldReconnect) {
        connectToWhatsApp();
      } else {
        console.log("CONNECTION_STATUS: BANNED");
      }
    } else if (connection === 'open') {
      console.log("CONNECTION_STATUS: CONNECTED");
      logger.info("Connection is now open.");
    }
  });
}

connectToWhatsApp();
