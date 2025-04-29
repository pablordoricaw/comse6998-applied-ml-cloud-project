# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-storage",
# ]
# ///
import argparse

from google.cloud import storage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self contained Python module to create (if not exists) the Google Cloud Storage bucket to act as backend and store the IaC state."
    )

    parser.add_argument(
        "--users",
        type=str,
        nargs="*",  # allow no users
        help="Google Cloud users to grant access to the bucket to manage infra.",
    )

    parser.add_argument("--clean", help="Delete bucket and user permissions")
    args = parser.parse_args()

    st_client = storage.Client()

    bkt = storage.Bucket(st_client, "bkt-infra-state")

    if not bkt.exists():
        bkt.storage_class = "STANDARD"

        bkt = st_client.create_bucket(bkt, location="us-east1")

        bkt.iam_configuration.uniform_bucket_level_access_enabled = True
        bkt.patch()

        print(f"Created bucket {bkt.name} with:")
        print(f" - Location: {bkt.location}")
        print(f" - Storage Class: {bkt.storage_class}")
        print(" - Uniform Bucket-Level Access: Enabled")

    else:
        print(f"Bucket {bkt.name} already exists.")

    if args.users:
        print("Manage user access to bucket")

        target_role = "roles/storage.objectUser"
        policy = bkt.get_iam_policy(requested_policy_version=3)
        role_binding = next(
            (b for b in policy.bindings if b["role"] == target_role),
            None,
        )

        for user in args.users:
            gcp_user = "user:" + user
            if role_binding:
                if gcp_user in role_binding["members"]:
                    print(f" - {user} already has access to {bkt.name}")
                    continue
                else:
                    role_binding["members"].add(gcp_user)
            else:
                policy.bindings.append(
                    {"role": target_role, "members": {gcp_user}, "condition": None}
                )

            bkt.set_iam_policy(policy)
            print(f" - Granted {user} access to {bkt.name}")
