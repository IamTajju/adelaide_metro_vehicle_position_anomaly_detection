import org.apache.flink.streaming.api.functions.source.legacy.SourceFunction;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class PollingVehicleSource implements SourceFunction<String> {

    private volatile boolean running = true;

    @Override
    public void run(SourceContext<String> ctx) throws Exception {
        while (running) {
            try {
                URL url = new URL("https://gtfs.adelaidemetro.com.au/v1/realtime/vehicle_positions");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(10000);
                conn.setReadTimeout(10000);

                BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                StringBuilder response = new StringBuilder();
                String inputLine;

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();

                ctx.collect(response.toString());
            } catch (Exception e) {
                ctx.collect("Error: " + e.getMessage());
            }

            Thread.sleep(15000);
        }
    }

    @Override
    public void cancel() {
        running = false;
    }
}