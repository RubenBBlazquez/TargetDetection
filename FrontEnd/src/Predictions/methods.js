export function getPredictions() {
    return new Promise((resolve, reject) => {
        const API_URL = process.env.REACT_APP_API_URL;

        fetch(`${API_URL}/get_all_predictions/`)
            .then(async (response) => {
                const data = await response.json();

                if (response.ok) {
                    return resolve(data);
                }

                return reject(data);
            })
            .catch((error) => {
                return reject(error);
            });
    })
}